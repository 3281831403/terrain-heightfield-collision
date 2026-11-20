"""Terrain height-field sampling and collision helpers.

World units are **meters** on every axis, and every floating point contract in
this module is stated with the tolerance ``TOL = 1e-6`` (meters, or unit-length
components for normalized vectors).

A height field is a plain ``dict``::

    {
        "origin": [x0, z0],           # world position of heights[0][0], meters
        "spacing": 0.5,               # uniform grid spacing in meters, must be > 0
        "heights": [[h, ...], ...],   # heights[iz][ix], meters, rectangular
    }

The sample ``heights[iz][ix]`` sits at the world position
``(x0 + ix * spacing, z0 + iz * spacing)``.  ``heights`` is indexed
``[row_z][column_x]``; every row must have the same length and the grid must be
non-empty.  Heights between samples come from bilinear interpolation, positions
outside the grid are clamped to the nearest edge sample, and a field whose
``spacing`` is zero or negative is invalid.

All functions raise :class:`TerrainError` on an invalid field; they never leak
``IndexError``, ``ZeroDivisionError`` or ``TypeError`` for bad input.
"""

from __future__ import annotations

import math

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

#: Absolute tolerance used for every floating point comparison in this module.
TOL = 1e-6


class TerrainError(ValueError):
    """Raised when a height field, body or config violates the documented contract."""


def validate_field(field) -> list[str]:
    """Return a list of human readable problems found in ``field``.

    The check is exhaustive and side-effect free; it never raises for a
    malformed field.  An empty list means the field is usable.  At minimum the
    following must be reported:

    * ``heights`` is empty, or is not a list of rows;
    * ``spacing`` is missing, non-numeric, or ``<= 0`` (a grid may not collapse);
    * the rows are not rectangular (ragged row lengths) - this is what keeps
      every later index computation in bounds;
    * ``origin`` is not a pair of finite numbers, or any height is not a finite
      number.

    Args:
        field: height field mapping as described in the module docstring.

    Returns:
        list[str]: problems, empty when the field is valid.
    """
    raise NotImplementedError("validate_field is not implemented")


def sample_height(field, x, z) -> float:
    """Height in meters at the world position ``(x, z)`` by bilinear interpolation.

    The sample ``heights[iz][ix]`` lives at ``(x0 + ix * spacing,
    z0 + iz * spacing)``.  ``(x, z)`` is located inside the surrounding cell and
    the four corner heights are blended with the fractional cell coordinates, so
    a linear ramp is reproduced exactly (``abs(error) <= TOL``) and a slope never
    degrades into a staircase.

    Positions outside ``[x0, x_max] x [z0, z_max]`` are clamped to the nearest
    edge sample: sampling one meter outside the grid returns the border height,
    it does not raise and it does not wrap or index out of bounds.

    Args:
        field: height field mapping as described in the module docstring.
        x: world x in meters, clamped to the grid extent.
        z: world z in meters, clamped to the grid extent.

    Returns:
        float: interpolated height in meters.

    Raises:
        TerrainError: if ``validate_field(field)`` is non-empty (for example a
            ``spacing`` of ``0`` would divide by zero inside the interpolation).
    """
    raise NotImplementedError("sample_height is not implemented")


def sample_normal(field, x, z) -> list[float]:
    """Unit surface normal in meters-space at ``(x, z)``.

    The gradient is taken from the neighbouring samples - a central difference
    inside the grid, a one-sided difference on the border - and divided by
    ``spacing`` so grid spacing never changes the result.  The normal is
    ``normalize((-dh/dx, 1.0, -dh/dz))``: it points up, and for a flat field it
    is exactly ``[0.0, 1.0, 0.0]``.

    The returned vector has length ``1.0`` within ``TOL``; shading and sliding
    code multiplies by it, so an unnormalized ``(dh/dx, 1, dh/dz)`` is wrong.

    Args:
        field: height field mapping as described in the module docstring.
        x: world x in meters.
        z: world z in meters.

    Returns:
        list[float]: ``[nx, ny, nz]``, unit length, ``ny > 0``.

    Raises:
        TerrainError: if ``validate_field(field)`` is non-empty.
    """
    raise NotImplementedError("sample_normal is not implemented")


def slope_deg(field, x, z) -> float:
    """Slope angle in degrees at ``(x, z)``, used by walkability checks.

    The angle is measured between the surface normal and the up axis, i.e.
    ``degrees(atan(sqrt((dh/dx)**2 + (dh/dz)**2)))``.  The height difference is
    divided by the horizontal distance (``spacing``), so the same 1 m rise over
    0.5 m and over 2 m of ground produce different angles, and a flat field
    returns ``0.0``.

    Args:
        field: height field mapping as described in the module docstring.
        x: world x in meters.
        z: world z in meters.

    Returns:
        float: slope angle in degrees, in ``[0.0, 90.0)``.

    Raises:
        TerrainError: if ``validate_field(field)`` is non-empty.
    """
    raise NotImplementedError("slope_deg is not implemented")


def raycast(field, origin, dir, max_dist) -> dict:
    """Cast a ray against the terrain surface.

    The ray is ``origin + t * normalize(dir)`` for ``t`` in ``[0, max_dist]``;
    ``dir`` need not be normalized and must not be the zero vector.  A coarse
    march along the ray (step of at most half a cell) finds a bracket where the
    ray passes from above to below the surface, and the bracket is then refined
    by bisection until it is narrower than ``1e-9`` meters.  The returned point
    therefore lies on the sampled surface within ``TOL`` instead of being
    reported half a cell early.

    Args:
        field: height field mapping as described in the module docstring.
        origin: ``[x, y, z]`` ray origin in meters.
        dir: ``[dx, dy, dz]`` ray direction, any non-zero length.
        max_dist: maximum travel distance in meters along the ray, ``>= 0``.

    Returns:
        dict: ``{"hit": bool, "t": float, "point": [x, y, z]}`` where ``t`` is
        the distance along the normalized direction and ``point`` is the hit
        position.  On a miss ``hit`` is ``False``, ``t`` equals ``max_dist`` and
        ``point`` is the end of the ray.

    Raises:
        TerrainError: if ``validate_field(field)`` is non-empty, or ``origin``,
            ``dir``, ``max_dist`` are malformed (zero-length ``dir``, negative
            ``max_dist``).
    """
    raise NotImplementedError("raycast is not implemented")


def resolve_sphere(field, center, radius) -> dict:
    """Push a sphere out of the terrain and report the corrected center.

    ``center`` is a ``[x, y, z]`` point in meters and ``radius`` is a sphere
    radius in meters (``> 0``).  When the sphere overlaps the terrain the center
    is moved **along the local surface normal** (:func:`sample_normal`) until the
    distance from the center to the surface is ``radius`` within ``TOL`` - a
    vertical-only lift keeps the sphere stuck on a slope and must not be used.
    A sphere that already clears the surface is returned unchanged.

    Args:
        field: height field mapping as described in the module docstring.
        center: ``[x, y, z]`` sphere center in meters.
        radius: sphere radius in meters, ``> 0``.

    Returns:
        dict: ``{"center": [x, y, z], "penetration": float, "corrected": bool}``.
        ``penetration`` is the overlap depth in meters that was removed (``0.0``
        when nothing moved) and ``corrected`` tells whether the center changed
        by more than ``TOL``.

    Raises:
        TerrainError: if ``validate_field(field)`` is non-empty, or ``center``
            is malformed or ``radius`` is negative.
    """
    raise NotImplementedError("resolve_sphere is not implemented")


def height_over_segment(field, a, b, steps) -> list[float]:
    """Sample the terrain height along a ground segment, for gluing to ground.

    ``a`` and ``b`` are ``[x, z]`` world endpoints in meters.  The segment is
    split into ``steps`` equal pieces and the height is sampled at every split
    point including both endpoints, so the result always has ``steps + 1``
    entries.  Split points are uniform in the horizontal plane - the subdivision
    is driven by arc length over the ground plane, not by a fixed number of grid
    cells, so a steep segment is not sampled more densely than a flat one of the
    same length.

    Args:
        field: height field mapping as described in the module docstring.
        a: ``[x, z]`` first endpoint in meters.
        b: ``[x, z]`` second endpoint in meters.
        steps: number of subdivisions, an integer ``>= 1``.

    Returns:
        list[float]: ``steps + 1`` heights in meters, ordered from ``a`` to ``b``.

    Raises:
        TerrainError: if ``validate_field(field)`` is non-empty, or ``a``, ``b``,
            ``steps`` are malformed (``steps < 1``, non-finite endpoints).
    """
    raise NotImplementedError("height_over_segment is not implemented")


def collide_batch(field, bodies, config) -> dict:
    """Resolve every body in ``bodies`` against ``field`` atomically.

    ``bodies`` is a list of ``{"id": str, "center": [x, y, z], "radius": float}``
    mappings; ``config`` is a mapping that may carry a default ``"radius"`` and a
    ``"max_dist"``-style tolerance, and may be empty.  Each body is resolved with
    :func:`resolve_sphere` using its **own** position and its **own** radius: no
    sampled height, normal or cell index may be shared between bodies, otherwise
    two bodies in the same grid cell inherit each other's terrain data.

    The call is all-or-nothing.  If a single body is invalid (missing or
    non-finite ``center``, non-positive ``radius``) nothing is applied: the
    result has ``ok`` ``False``, an empty ``bodies`` list and an ``error``
    message, and the input list is left untouched.

    Args:
        field: height field mapping as described in the module docstring.
        bodies: list of body mappings as described above.
        config: mapping with optional defaults; ``{}`` is valid.

    Returns:
        dict: ``{"ok": bool, "bodies": list[dict], "error": str | None}``.
        On success ``ok`` is ``True``, ``error`` is ``None`` and ``bodies`` holds
        one entry per input body **in input order**, each with the input ``id``
        and ``radius`` plus the resolved ``center`` and ``corrected`` flag.

    Raises:
        TerrainError: if ``validate_field(field)`` is non-empty, or ``bodies`` /
            ``config`` are not mappings in the expected shape.  Invalid bodies
            are reported through the ``ok``/``error`` result, not by raising.
    """
    raise NotImplementedError("collide_batch is not implemented")
