"""Terrain height-field collision package."""

from .collision import (
    TOL,
    TerrainError,
    collide_batch,
    height_over_segment,
    raycast,
    resolve_sphere,
    sample_height,
    sample_normal,
    slope_deg,
    validate_field,
)

__all__ = [
    "TOL",
    "TerrainError",
    "collide_batch",
    "height_over_segment",
    "raycast",
    "resolve_sphere",
    "sample_height",
    "sample_normal",
    "slope_deg",
    "validate_field",
]
