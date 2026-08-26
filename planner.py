"""CrowdSurfer (VQ-VAE + PixelCNN + PRIEST) wrapper for the arena_planners bridge.

CrowdSurfer (ICRA 2025, Smart-Wheelchair-RRC) does dense-crowd navigation by
generating candidate trajectories with a VQ-VAE + PixelCNN and then selecting /
refining one with a PRIEST sampling optimizer. This wrapper reconstructs the
upstream ``LivePipeline`` observation outside their hydra/ROS stack and maps the
result to a differential-drive ``[v, omega]`` twist.

Observation reconstruction (all ego frame), mirroring
``inference/live.py`` + ``ros_interface.py``:
  * static_obstacles : a 60x60 occupancy map rasterized from the 2D LaserScan
    (ranges + reconstructed angles), ego at the centre — _generate_occupancy_map.
  * dynamic_obstacles : last 5 timesteps of [x, y, vx, vy] for up to 10 nearest
    pedestrians, in the robot frame, padded far away (1000) when absent.
  * heading_to_goal   : atan2 of the goal in the robot frame.
  * projection inputs : ego velocity/acceleration (held from the previous control
    cycle's trajectory, as upstream does), goal position, and the obstacle
    position/velocity tables PRIEST consumes.

The control command is extracted from the PRIEST-selected Bernstein trajectory
exactly as ``ros_interface.compute_controls`` / ``plan`` do: average the first
few first-differential samples for (vx, vy), then project onto the heading to get
forward velocity and turn rate. Device-agnostic; runs on CPU.

Attribution: network + PRIEST/projection code vendored under crowdsurfer_net/
from https://github.com/Smart-Wheelchair-RRC/CrowdSurfer (priest_core JIT
annotations de-GPU'd for CPU execution; configuration reduced to two enums).
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import torch

from crowdsurfer_net.inference import (
    _MAP_HEIGHT,
    _MAP_RESOLUTION,
    _MAP_WIDTH,
    _MAX_DYNAMIC_OBSTACLES,
    _MAX_STATIC_OBSTACLES,
    _OBSTACLE_PADDING,
    _TRAJECTORY_TIME,
    CrowdSurferPipeline,
    InferenceData,
)

_HERE = pathlib.Path(__file__).parent
_CHECKPOINT_DIR = os.path.join(str(_HERE), "model", "crowdsurfer_best_64_4")

# Lidar field of view used to reconstruct beam angles from the ranges vector.
# The bridge LaserScanCollector forwards only the ranges array, so we rebuild the
# angles as linspace over the FOV. Arena lidars are 360-deg; matches the smoke
# harness (angle_min=-pi, angle_max=pi).
_SCAN_ANGLE_MIN = -np.pi
_SCAN_MAX_USEFUL_RANGE = 50.0

# Control extraction (ros_interface.py): average the first num_control_samples
# first-differential samples; scale coefficients by 0.8 before differentiating.
_NUM_CONTROL_SAMPLES = 6
_COEFF_SCALE = 0.8
# desiredVelocity stop gate (ros_interface._check_dynamic_obstacle_distance_for_stopping).
_STOP_DISTANCE = 1.2

# Jackal limits (control.yaml: linear.x 2.0, angular.z 4.0). The upstream omega
# law (-zeta/0.3) is wheelchair-tuned and reaches ~10 rad/s; clamp below the cap
# to keep steering trackable.
_MAX_V = 2.0
_MAX_OMEGA = 2.0

_NUM_PREVIOUS_TIMESTEPS = 5

_pipeline: CrowdSurferPipeline | None = None


def _get_pipeline() -> CrowdSurferPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CrowdSurferPipeline(checkpoint_dir=_CHECKPOINT_DIR, device="cpu")
    return _pipeline


# Rolling history of dynamic obstacles in the robot frame: (5, 4, max_dyn).
# Channels [x, y, vx, vy]; positions padded far away (1000), velocities zero.
def _empty_dynamic_history() -> np.ndarray:
    history = np.concatenate(
        (
            np.full((_NUM_PREVIOUS_TIMESTEPS, 2, _MAX_DYNAMIC_OBSTACLES), _OBSTACLE_PADDING, dtype=np.float32),
            np.zeros((_NUM_PREVIOUS_TIMESTEPS, 2, _MAX_DYNAMIC_OBSTACLES), dtype=np.float32),
        ),
        axis=1,
    )
    return history


_dynamic_history: np.ndarray = _empty_dynamic_history()
# Ego velocity / acceleration carried across control cycles (upstream feeds the
# previous cycle's commanded controls back into the next projection step).
_ego_velocity = np.zeros(2, dtype=np.float32)
_ego_acceleration = np.zeros(2, dtype=np.float32)


def _generate_occupancy_map(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rasterize a 2D scan into a 60x60 occupancy map (LivePipeline logic)."""
    occupancy_map = np.zeros((_MAP_HEIGHT, _MAP_WIDTH), dtype=np.float32)
    origin = (_MAP_WIDTH * _MAP_RESOLUTION / 2, _MAP_HEIGHT * _MAP_RESOLUTION / 2)

    valid = ~np.isnan(ranges) & ~np.isnan(angles)
    valid_ranges = ranges[valid]
    valid_angles = angles[valid]
    if valid_ranges.size != 0:
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        map_x = np.round((x + origin[0]) / _MAP_RESOLUTION).astype(int)
        map_y = np.round((y + origin[1]) / _MAP_RESOLUTION).astype(int)
        inside = (map_x >= 0) & (map_x < _MAP_WIDTH) & (map_y >= 0) & (map_y < _MAP_HEIGHT)
        map_x, map_y = map_x[inside], map_y[inside]
        if map_x.size > 0:
            occupancy_map[map_y, map_x] = 100.0  # occupied
        occupancy_map[_MAP_HEIGHT // 2, _MAP_WIDTH // 2] = 50.0  # ego placeholder
    return occupancy_map


def _scan_to_ego_points(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Cartesian ego-frame (2, N) static points from a valid scan, sorted near->far."""
    valid = ~np.isnan(ranges) & ~np.isnan(angles) & (ranges <= _SCAN_MAX_USEFUL_RANGE)
    r = ranges[valid]
    a = angles[valid]
    if r.size == 0:
        return np.zeros((2, 0), dtype=np.float32)
    pts = np.stack((r * np.cos(a), r * np.sin(a)))  # (2, N)
    order = np.argsort(np.linalg.norm(pts, axis=0))
    return pts[:, order]


def _update_dynamic_history(pedestrians, robot_pose) -> None:
    """Shift the rolling history and push the latest pedestrians (robot frame)."""
    global _dynamic_history
    _dynamic_history[:-1] = _dynamic_history[1:]
    _dynamic_history[-1, :2, :] = _OBSTACLE_PADDING
    _dynamic_history[-1, 2:, :] = 0.0

    if pedestrians is None or len(pedestrians) == 0 or robot_pose is None or len(robot_pose) < 3:
        return

    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    rows = []
    for ped in pedestrians:
        # ArenaPedestrianCollector rows: [id, x, y, vx, vy] in world frame.
        wx, wy, wvx, wvy = float(ped[1]), float(ped[2]), float(ped[3]), float(ped[4])
        dx, dy = wx - px, wy - py
        ex = cos_t * dx + sin_t * dy
        ey = -sin_t * dx + cos_t * dy
        evx = cos_t * wvx + sin_t * wvy
        evy = -sin_t * wvx + cos_t * wvy
        rows.append((ex * ex + ey * ey, ex, ey, evx, evy))
    rows.sort(key=lambda r: r[0])  # nearest first
    for i in range(min(_MAX_DYNAMIC_OBSTACLES, len(rows))):
        _, ex, ey, evx, evy = rows[i]
        _dynamic_history[-1, 0, i] = ex
        _dynamic_history[-1, 1, i] = ey
        _dynamic_history[-1, 2, i] = evx
        _dynamic_history[-1, 3, i] = evy


def _goal_in_robot_frame(robot_pose, goal_pose) -> np.ndarray:
    """Goal (x, y) expressed in the robot frame; (5, 0) fallback if unavailable."""
    if robot_pose is None or goal_pose is None or len(robot_pose) < 3 or len(goal_pose) < 2:
        return np.array([5.0, 0.0], dtype=np.float32)
    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    gx, gy = float(goal_pose[0]), float(goal_pose[1])
    dx, dy = gx - px, gy - py
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    return np.array([cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy], dtype=np.float32)


def _build_projection_obstacles() -> tuple[np.ndarray, np.ndarray]:
    """Obstacle position/velocity tables for PRIEST: (2, max_dyn + max_static)."""
    total = _MAX_DYNAMIC_OBSTACLES + _MAX_STATIC_OBSTACLES
    positions = np.full((2, total), _OBSTACLE_PADDING, dtype=np.float32)
    velocities = np.zeros((2, total), dtype=np.float32)
    latest = _dynamic_history[-1]  # (4, max_dyn)
    positions[:, :_MAX_DYNAMIC_OBSTACLES] = latest[:2, :]
    velocities[:, :_MAX_DYNAMIC_OBSTACLES] = latest[2:, :]
    return positions, velocities


def _add_static_points_to_projection(positions: np.ndarray, static_points: np.ndarray) -> None:
    """Fill the static slots of the projection table with nearest scan points."""
    n = min(_MAX_STATIC_OBSTACLES, static_points.shape[1])
    if n > 0:
        positions[:, _MAX_DYNAMIC_OBSTACLES : _MAX_DYNAMIC_OBSTACLES + n] = static_points[:, :n]


def _extract_controls(best_coefficients: np.ndarray) -> tuple[float, float]:
    """Bernstein-trajectory -> (v, omega) (ros_interface.compute_controls/plan)."""
    pipeline = _get_pipeline()
    x_best = best_coefficients[0] * _COEFF_SCALE
    y_best = best_coefficients[1] * _COEFF_SCALE

    xdot = pipeline.bernstein_first_diff @ x_best
    ydot = pipeline.bernstein_first_diff @ y_best
    xddot = pipeline.bernstein_second_diff @ x_best
    yddot = pipeline.bernstein_second_diff @ y_best

    vx = float(np.mean(xdot[:_NUM_CONTROL_SAMPLES]))
    vy = float(np.mean(ydot[:_NUM_CONTROL_SAMPLES]))
    ax = float(np.mean(xddot[:_NUM_CONTROL_SAMPLES]))
    ay = float(np.mean(yddot[:_NUM_CONTROL_SAMPLES]))

    global _ego_velocity, _ego_acceleration
    _ego_velocity = np.array((vx, vy), dtype=np.float32)
    _ego_acceleration = np.array((ax, ay), dtype=np.float32)

    norm_v = float(np.hypot(vx, vy))
    angle_v = float(np.arctan2(vy, vx))
    # Trajectory is already in the robot frame, so the current heading is 0.
    zeta = 0.0 - angle_v
    v = norm_v * np.cos(zeta)
    omega = -zeta / (_NUM_CONTROL_SAMPLES * _TRAJECTORY_TIME * 0.01)
    v = float(np.clip(v, -_MAX_V, _MAX_V))
    omega = float(np.clip(omega, -_MAX_OMEGA, _MAX_OMEGA))
    return v, omega


def _nearest_dynamic_obstacle_distance() -> float:
    latest = _dynamic_history[-1, :2, :]  # (2, max_dyn)
    valid = latest[:, (latest < _OBSTACLE_PADDING).all(axis=0)]
    if valid.shape[1] == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(valid, axis=0)))


def step(features: dict) -> list[float]:
    """Map the bridge feature dict to a differential-drive [v, omega] twist."""
    pipeline = _get_pipeline()

    robot_pose = features.get("robot_pose")
    goal_pose = features.get("goal_pose")
    pedestrians = features.get("pedestrians")

    # --- static obstacles: occupancy map from the 2D scan -----------------
    scan_raw = features.get("laser_scan")
    if scan_raw is not None and len(scan_raw) > 0:
        ranges = np.asarray(scan_raw, dtype=np.float32)
        angles = (_SCAN_ANGLE_MIN + 2.0 * np.pi * np.arange(len(ranges)) / len(ranges)).astype(np.float32)
    else:
        ranges = np.zeros(0, dtype=np.float32)
        angles = np.zeros(0, dtype=np.float32)
    occupancy_map = _generate_occupancy_map(ranges, angles)
    static_points = _scan_to_ego_points(ranges, angles)

    # --- dynamic obstacles: rolling pedestrian history --------------------
    _update_dynamic_history(pedestrians, robot_pose)

    # --- goal / heading ---------------------------------------------------
    goal = _goal_in_robot_frame(robot_pose, goal_pose)
    heading_to_goal = float(np.arctan2(goal[1], goal[0]))

    # --- projection obstacle tables (PRIEST) ------------------------------
    obstacle_positions, obstacle_velocities = _build_projection_obstacles()
    _add_static_points_to_projection(obstacle_positions, static_points)

    device = pipeline.device
    data = InferenceData(
        static_obstacles=torch.from_numpy(occupancy_map).unsqueeze(0).unsqueeze(0).float().to(device),
        dynamic_obstacles=torch.from_numpy(_dynamic_history).unsqueeze(0).float().to(device),
        heading_to_goal=torch.tensor([heading_to_goal], dtype=torch.float32, device=device),
        ego_velocity_for_projection=torch.from_numpy(_ego_velocity).unsqueeze(0).float().to(device),
        ego_acceleration_for_projection=torch.from_numpy(_ego_acceleration).unsqueeze(0).float().to(device),
        goal_position_for_projection=torch.from_numpy(goal).unsqueeze(0).float().to(device),
        obstacle_positions_for_projection=torch.from_numpy(obstacle_positions).unsqueeze(0).float().to(device),
        obstacle_velocities_for_projection=torch.from_numpy(obstacle_velocities).unsqueeze(0).float().to(device),
    )

    best_coefficients = pipeline.plan(data)  # (2, 11)
    v, omega = _extract_controls(best_coefficients)

    # Stop gate: a pedestrian inside the threshold -> rotate in place.
    if _nearest_dynamic_obstacle_distance() <= _STOP_DISTANCE:
        return [0.0, omega * 1.5]
    return [v, omega]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _dynamic_history, _ego_velocity, _ego_acceleration
    _dynamic_history = _empty_dynamic_history()
    _ego_velocity = np.zeros(2, dtype=np.float32)
    _ego_acceleration = np.zeros(2, dtype=np.float32)


if __name__ == "__main__":
    from arena_planners.sdk import load_manifest, main_loop

    manifest = load_manifest(_HERE / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
