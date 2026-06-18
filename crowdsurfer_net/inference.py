"""Self-contained CrowdSurfer inference pipeline (one control step).

Reconstructs, outside the upstream hydra/ROS stack, the ``LivePipeline`` from
``src/CrowdSurfer/inference/{pipeline,live}.py``:

  build obs  ->  PixelCNN(obs) -> codebook log-probs + observation embedding
             ->  multinomial sample of codebook indices
             ->  VQ-VAE.decode_from_indices -> candidate Bernstein coefficients
             ->  PRIEST optimization selects/refines the elite coefficients
             ->  best trajectory -> immediate velocity command

The shipped checkpoints are ``vqvae_best_64_4.bin`` + ``pixelcnn_best_64_4.bin``
(num_embeddings=64, embedding_dim=4). No scoring-network checkpoint is published,
so — exactly as upstream's ``_run_pipeline`` does when ``scoring_network is
None`` — trajectory selection is delegated to PRIEST's ``idx_min``. PRIEST is
therefore load-bearing, not optional: dropping it would remove the only selector
and change behavior.

Everything runs on CPU (device-agnostic): the vendored ``priest_core`` has its
hardcoded ``backend="gpu"`` JIT annotations stripped so JAX uses the default
device, and torch tensors follow ``self.device``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from crowdsurfer_net.configuration import StaticObstacleType
from crowdsurfer_net.navigation.model import VQVAE, CombinedPixelCNN
from crowdsurfer_net.navigation.projection_guidance import ProjectionGuidance
from crowdsurfer_net.navigation.priest_guidance import PriestPlanner

# --- Configuration mirrored from upstream defaults -------------------------
# configuration/configuration.py defaults + base_vqvae(64,4,96), base_pixelcnn(32).
_NUM_EMBEDDINGS = 64
_EMBEDDING_DIM = 4
_VQVAE_HIDDEN_CHANNELS = 96
_OBSERVATION_EMBEDDING_DIM = 32

# DatasetConfiguration / ProjectionConfiguration / ScoringNetworkConfiguration.
_TRAJECTORY_LENGTH = 50          # dataset.trajectory_length
_TRAJECTORY_TIME = 5             # dataset.trajectory_time (seconds)
_NUM_SAMPLES = 50               # scoring_network.num_samples (CEM / sampling batch)
_MAX_DYNAMIC_OBSTACLES = 10      # projection.max_dynamic_obstacles
_MAX_STATIC_OBSTACLES = 100      # projection.max_static_obstacles
_OBSTACLE_PADDING = 1000.0       # projection.padding (far-away sentinel)
_ROBOT_RADIUS = 0.5             # projection.robot_radius
_OBSTACLE_RADIUS = 0.3          # projection.obstacle_radius
_TRACKING_WEIGHT = 1.0          # projection.tracking_weight
_SMOOTHNESS_WEIGHT = 0.2        # projection.smoothness_weight
_MAX_OUTER_ITERATIONS = 2       # projection.max_outer_iterations
_MAX_INNER_ITERATIONS = 13      # projection.max_inner_iterations
_MAX_VELOCITY = 1.0             # projection.max_velocity
_DESIRED_VELOCITY = 1.0         # projection.desired_velocity

# Occupancy-map geometry (LivePipeline._generate_occupancy_map defaults).
_MAP_HEIGHT = 60
_MAP_WIDTH = 60
_MAP_RESOLUTION = 0.1


@dataclass
class InferenceData:
    """Ego-frame observation for one control step (see upstream pipeline.py)."""

    static_obstacles: Tensor          # (1, 1, H, W) occupancy map
    dynamic_obstacles: Tensor         # (1, 5, 4, max_dyn) [x, y, vx, vy] over 5 steps
    heading_to_goal: Tensor           # (1,) radians
    ego_velocity_for_projection: Tensor          # (1, 2) [vx, vy]
    ego_acceleration_for_projection: Tensor      # (1, 2) [ax, ay]
    goal_position_for_projection: Tensor          # (1, 2) [x, y]
    obstacle_positions_for_projection: Tensor     # (1, 2, max_dyn + max_static)
    obstacle_velocities_for_projection: Tensor    # (1, 2, max_dyn + max_static)


class CrowdSurferPipeline:
    """VQ-VAE + PixelCNN generation, PRIEST selection/refinement."""

    def __init__(self, checkpoint_dir: str, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.num_samples = _NUM_SAMPLES
        self.max_dynamic_obstacles = _MAX_DYNAMIC_OBSTACLES
        self.max_static_obstacles = _MAX_STATIC_OBSTACLES

        vqvae_path = os.path.join(checkpoint_dir, "vqvae_best_64_4.bin")
        pixelcnn_path = os.path.join(checkpoint_dir, "pixelcnn_best_64_4.bin")

        self.vqvae = VQVAE(
            num_embeddings=_NUM_EMBEDDINGS,
            embedding_dim=_EMBEDDING_DIM,
            hidden_channels=_VQVAE_HIDDEN_CHANNELS,
        ).to(self.device)
        self.pixelcnn = CombinedPixelCNN(
            num_embeddings=_NUM_EMBEDDINGS,
            vqvae_hidden_channels=_VQVAE_HIDDEN_CHANNELS,
            observation_embedding_dim=_OBSERVATION_EMBEDDING_DIM,
            static_obstacle_type=StaticObstacleType.OCCUPANCY_MAP,
        ).to(self.device)

        # The .bin checkpoints are plain state_dicts whose keys match the module
        # tree exactly; load strict to fail loud on any architecture drift.
        self.vqvae.load_state_dict(
            torch.load(vqvae_path, map_location=self.device), strict=True
        )
        self.pixelcnn.load_state_dict(
            torch.load(pixelcnn_path, map_location=self.device), strict=True
        )
        self.vqvae.eval()
        self.pixelcnn.eval()

        # Bernstein basis (and its derivatives) for coefficients -> trajectory and
        # the immediate-velocity control extraction. num_obstacles=0: no projection
        # guidance path is used here (PRIEST does the obstacle-aware optimization).
        inflation_radius = _OBSTACLE_RADIUS + _ROBOT_RADIUS
        self.projection_guidance = ProjectionGuidance(
            num_obstacles=0,
            num_timesteps=_TRAJECTORY_LENGTH,
            total_time=_TRAJECTORY_TIME,
            obstacle_ellipse_semi_major_axis=inflation_radius,
            obstacle_ellipse_semi_minor_axis=inflation_radius,
            max_projection_iterations=2,
            device=self.device,
        )

        self.priest_planner = PriestPlanner(
            num_dynamic_obstacles=_MAX_DYNAMIC_OBSTACLES,
            num_static_obstacles=_MAX_STATIC_OBSTACLES,
            time_horizon=_TRAJECTORY_TIME,
            trajectory_length=_TRAJECTORY_LENGTH,
            tracking_weight=_TRACKING_WEIGHT,
            smoothness_weight=_SMOOTHNESS_WEIGHT,
            static_obstacle_semi_minor_axis=inflation_radius,
            static_obstacle_semi_major_axis=inflation_radius,
            dynamic_obstacle_semi_minor_axis=inflation_radius,
            dynamic_obstacle_semi_major_axis=inflation_radius,
            num_waypoints=_TRAJECTORY_LENGTH,
            trajectory_batch_size=self.num_samples,
            max_outer_iterations=_MAX_OUTER_ITERATIONS,
            max_inner_iterations=_MAX_INNER_ITERATIONS,
            max_velocity=_MAX_VELOCITY,
            desired_velocity=_DESIRED_VELOCITY,
        )

        # Bernstein first/second differential matrices for control extraction
        # (ros_interface.compute_controls).
        self.bernstein_first_diff = (
            self.projection_guidance.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL.detach()
            .cpu()
            .numpy()
        )
        self.bernstein_second_diff = (
            self.projection_guidance.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL.detach()
            .cpu()
            .numpy()
        )

    # --- generation -------------------------------------------------------
    def _sample_coefficients(self, data: InferenceData) -> tuple[Tensor, Tensor]:
        """PixelCNN -> codebook distribution -> sample -> VQ-VAE decode."""
        with torch.no_grad():
            pixelcnn_output, observation_embedding = self.pixelcnn.forward(
                static_obstacles=data.static_obstacles,
                dynamic_obstacles=data.dynamic_obstacles,
                heading_to_goal=data.heading_to_goal,
            )
            probability_distribution = torch.exp(
                torch.log_softmax(pixelcnn_output, dim=-1)
            )  # (batch, hidden_channels, num_embeddings)

            pixelcnn_indices = (
                torch.multinomial(
                    probability_distribution.flatten(0, 1),
                    num_samples=self.num_samples,
                    replacement=True,
                )
                .unflatten(0, (-1, probability_distribution.shape[1]))
                .transpose(1, 2)
                .flatten(0, 1)
            )  # (batch * num_samples, hidden_channels)

            coefficients, _ = self.vqvae.decode_from_indices(pixelcnn_indices)

        return coefficients, observation_embedding  # (num_samples, 2, 11), (1, emb)

    # --- selection (PRIEST) ----------------------------------------------
    def _run_priest(self, coefficients: Tensor, data: InferenceData) -> tuple[Tensor, int]:
        obstacle_positions = data.obstacle_positions_for_projection[0].cpu().numpy()
        obstacle_velocities = (
            data.obstacle_velocities_for_projection[0, :, : self.max_dynamic_obstacles]
            .cpu()
            .numpy()
        )

        _, _, _, _, c_x_elite, c_y_elite, _, _, idx_min = (
            self.priest_planner.run_optimization(
                initial_x_position=0.0,
                initial_y_position=0.0,
                initial_x_velocity=float(data.ego_velocity_for_projection[0, 0]),
                initial_y_velocity=float(data.ego_velocity_for_projection[0, 1]),
                initial_x_acceleration=float(data.ego_acceleration_for_projection[0, 0]),
                initial_y_acceleration=float(data.ego_acceleration_for_projection[0, 1]),
                goal_x_position=float(data.goal_position_for_projection[0, 0]),
                goal_y_position=float(data.goal_position_for_projection[0, 1]),
                dynamic_obstacle_x_positions=obstacle_positions[0, : self.max_dynamic_obstacles],
                dynamic_obstacle_y_positions=obstacle_positions[1, : self.max_dynamic_obstacles],
                dynamic_obstacle_x_velocities=obstacle_velocities[0],
                dynamic_obstacle_y_velocities=obstacle_velocities[1],
                static_obstacle_x_positions=obstacle_positions[0, self.max_dynamic_obstacles :],
                static_obstacle_y_positions=obstacle_positions[1, self.max_dynamic_obstacles :],
                custom_x_coefficients=coefficients[:, 0, :].cpu().numpy(),
                custom_y_coefficients=coefficients[:, 1, :].cpu().numpy(),
            )
        )

        selected = torch.stack(
            (torch.tensor(np.array(c_x_elite)), torch.tensor(np.array(c_y_elite))),
            dim=1,
        ).to(self.device)  # (num_samples, 2, 11)
        return selected, int(idx_min)

    # --- public step ------------------------------------------------------
    def plan(self, data: InferenceData) -> np.ndarray:
        """Return the best trajectory coefficients, shape (2, 11) [x, y]."""
        coefficients, _ = self._sample_coefficients(data)
        selected, idx_min = self._run_priest(coefficients, data)
        return selected[idx_min].cpu().numpy()  # (2, 11)
