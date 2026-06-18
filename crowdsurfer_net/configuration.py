"""Minimal configuration enums for the vendored CrowdSurfer navigation modules.

Upstream this lives in ``configuration/configuration.py`` and pulls in
hydra/omegaconf for the training/inference CLI. The inference network code only
needs these two enums, so we provide a dependency-free stand-in to avoid hauling
the whole config stack (and its deps) into the planner runtime.
"""

from __future__ import annotations

from enum import Enum, auto


class StaticObstacleType(Enum):
    OCCUPANCY_MAP = auto()
    POINT_CLOUD = auto()


class GuidanceType(Enum):
    PRIEST = auto()
    PROJECTION = auto()
