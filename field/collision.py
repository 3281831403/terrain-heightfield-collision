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
from collections.abc import Mapping

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

_MISSING = object()


class TerrainError(ValueError):
    """Raised when a height field, body or config violates the documented contract."""


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------


def _is_finite_number(value) -> bool:
    """True for real, finite numbers (``bool`` is deliberately excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _finite(value, name: str) -> float:
    if not _is_finite_number(value):
        raise TerrainError(f"{name} must be a finite number")
    return float(value)


def _vector(value, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise TerrainError(f"{name} must be a sequence of {length} finite numbers")
    return [_finite(component, f"{name}[{index}]") for index, component in enumerate(value)]


def _require_valid(field) -> None:
    problems = validate_field(field)
    if problems:
        raise TerrainError("invalid height field: " + "; ".join(problems))


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
    if not isinstance(field, Mapping):
        return ["field must be a mapping"]

    problems: list[str] = []

    spacing = field.get("spacing", _MISSING)
    if spacing is _MISSING:
        problems.append("spacing is missing")
    elif not _is_finite_number(spacing):
        problems.append("spacing must be a finite number")
    elif float(spacing) <= 0.0:
        problems.append("spacing must be greater than zero")

    origin = field.get("origin", _MISSING)
    if origin is _MISSING:
        problems.append("origin is missing")
    elif not isinstance(origin, (list, tuple)) or len(origin) != 2:
        problems.append("origin must be a pair of finite numbers [x, z]")
    else:
        for index, value in enumerate(origin):
            if not _is_finite_number(value):
                problems.append(f"origin[{index}] must be a finite number")

    heights = field.get("heights", _MISSING)
    if heights is _MISSING:
        problems.append("heights is missing")
    elif not isinstance(heights, (list, tuple)):
        problems.append("heights must be a list of rows")
    elif len(heights) == 0:
        problems.append("heights must not be empty")
    else:
        expected_len = None
        for iz, row in enumerate(heights):
            if not isinstance(row, (list, tuple)):
                problems.append(f"heights[{iz}] must be a list of numbers")
                continue
            if len(row) == 0:
                problems.append(f"heights[{iz}] must not be empty")
            if expected_len is None:
                expected_len = len(row)
            elif len(row) != expected_len:
                problems.append(
                    f"heights[{iz}] has {len(row)} samples, expected {expected_len} (ragged grid)"
                )
            for ix, value in enumerate(row):
                if not _is_finite_number(value):
                    problems.append(f"heights[{iz}][{ix}] must be a finite number")

    return problems


# ---------------------------------------------------------------------------
# internal layout / sampling primitives (callers have validated already)
# ---------------------------------------------------------------------------


def _layout(field):
    """Validate ``field`` and return ``(x0, z0, spacing, heights, nx, nz)``."""
    _require_valid(field)
    heights = field["heights"]
    nx = len(heights[0])
    nz = len(heights)
    return (
        float(field["origin"][0]),
        float(field["origin"][1]),
        float(field["spacing"]),
        heights,
        nx,
        nz,
    )


def _nearest_index(grid_coord: float, count: int) -> int:
    """Nearest sample index, clamped to ``[0, count - 1]``."""
    index = int(math.floor(grid_coord + 0.5))
    if index < 0:
        return 0
    if index > count - 1:
        return count - 1
    return index


def _clamp_to_extent(world: float, origin: float, spacing: float, count: int) -> float:
    """Grid coordinate of ``world`` clamped to the sample extent."""
    grid_coord = (world - origin) / spacing
    maximum = float(count - 1)
    if grid_coord < 0.0:
        return 0.0
    if grid_coord > maximum:
        return maximum
    return grid_coord


def _sample_height(layout, x: float, z: float) -> float:
    """Bilinear height for an already validated ``layout``."""
    x0, z0, spacing, heights, nx, nz = layout

    gx = _clamp_to_extent(x, x0, spacing, nx)
    gz = _clamp_to_extent(z, z0, spacing, nz)

    if nx == 1:
        ix0 = ix1 = 0
        fx = 0.0
    else:
        ix0 = int(math.floor(gx))
        if ix0 > nx - 2:
            ix0 = nx - 2
        ix1 = ix0 + 1
        fx = gx - ix0

    if nz == 1:
        iz0 = iz1 = 0
        fz = 0.0
    else:
        iz0 = int(math.floor(gz))
        if iz0 > nz - 2:
            iz0 = nz - 2
        iz1 = iz0 + 1
        fz = gz - iz0

    row0 = heights[iz0]
    row1 = heights[iz1]
    h00 = float(row0[ix0])
    h10 = float(row0[ix1])
    h01 = float(row1[ix0])
    h11 = float(row1[ix1])

    return (
        h00 * (1.0 - fx) * (1.0 - fz)
        + h10 * fx * (1.0 - fz)
        + h01 * (1.0 - fx) * fz
        + h11 * fx * fz
    )


def _gradient(layout, x: float, z: float) -> tuple[float, float]:
    """``(dh/dx, dh/dz)`` in meters per meter for an already validated layout."""
    x0, z0, spacing, heights, nx, nz = layout

    gx = _clamp_to_extent(x, x0, spacing, nx)
    gz = _clamp_to_extent(z, z0, spacing, nz)
    ix = _nearest_index(gx, nx)
    iz = _nearest_index(gz, nz)

    if nx == 1:
        dh_dx = 0.0
    elif ix == 0:
        dh_dx = (heights[iz][1] - heights[iz][0]) / spacing
    elif ix == nx - 1:
        dh_dx = (heights[iz][nx - 1] - heights[iz][nx - 2]) / spacing
    else:
        dh_dx = (heights[iz][ix + 1] - heights[iz][ix - 1]) / (2.0 * spacing)

    if nz == 1:
        dh_dz = 0.0
    elif iz == 0:
        dh_dz = (heights[1][ix] - heights[0][ix]) / spacing
    elif iz == nz - 1:
        dh_dz = (heights[nz - 1][ix] - heights[nz - 2][ix]) / spacing
    else:
        dh_dz = (heights[iz + 1][ix] - heights[iz - 1][ix]) / (2.0 * spacing)

    return dh_dx, dh_dz


def _sample_normal(layout, x: float, z: float) -> list[float]:
    """Unit normal for an already validated layout."""
    dh_dx, dh_dz = _gradient(layout, x, z)
    length = math.sqrt(dh_dx * dh_dx + 1.0 + dh_dz * dh_dz)
    return [-dh_dx / length, 1.0 / length, -dh_dz / length]


def _resolve_sphere(layout, center, radius: float) -> dict:
    """Resolve a validated sphere against an already validated layout."""
    original = [float(center[0]), float(center[1]), float(center[2])]

    normal = _sample_normal(layout, original[0], original[2])
    height = _sample_height(layout, original[0], original[2])
    distance = (original[1] - height) * normal[1]

    if distance >= radius - TOL:
        return {"center": original, "penetration": 0.0, "corrected": False}

    penetration = max(0.0, radius - distance)
    current = list(original)

    # Moving along the surface normal shifts x/z too, which changes the sampled
    # height underneath the sphere; repeat until the perpendicular distance is
    # the radius.  On flat or planar ground the first step is already exact.
    for _ in range(64):
        normal = _sample_normal(layout, current[0], current[2])
        height = _sample_height(layout, current[0], current[2])
        distance = (current[1] - height) * normal[1]
        gap = radius - distance
        if abs(gap) <= TOL:
            break
        current = [
            current[0] + gap * normal[0],
            current[1] + gap * normal[1],
            current[2] + gap * normal[2],
        ]

    moved = math.dist(original, current)
    return {
        "center": current,
        "penetration": penetration,
        "corrected": moved > TOL,
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


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
    layout = _layout(field)
    return _sample_height(layout, _finite(x, "x"), _finite(z, "z"))


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
    layout = _layout(field)
    return _sample_normal(layout, _finite(x, "x"), _finite(z, "z"))


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
    layout = _layout(field)
    dh_dx, dh_dz = _gradient(layout, _finite(x, "x"), _finite(z, "z"))
    return math.degrees(math.atan(math.hypot(dh_dx, dh_dz)))


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
    layout = _layout(field)
    ox, oy, oz = _vector(origin, 3, "origin")
    dx, dy, dz = _vector(dir, 3, "dir")
    max_dist = _finite(max_dist, "max_dist")

    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0.0:
        raise TerrainError("dir must not be the zero vector")
    if max_dist < 0.0:
        raise TerrainError("max_dist must not be negative")

    dx /= length
    dy /= length
    dz /= length

    def end_point(distance: float) -> list[float]:
        return [ox + distance * dx, oy + distance * dy, oz + distance * dz]

    def gap(distance: float) -> float:
        """Vertical distance from the ray point to the terrain (positive = above)."""
        px, py, pz = end_point(distance)
        return py - _sample_height(layout, px, pz)

    # Starting on or below the surface already intersects it at t = 0.
    if gap(0.0) <= 0.0:
        return {"hit": True, "t": 0.0, "point": end_point(0.0)}

    spacing = layout[2]
    step = 0.5 * spacing
    count = int(math.ceil(max_dist / step)) if max_dist > 0.0 else 0

    lo = 0.0
    hi = None
    for index in range(1, count + 1):
        distance = min(index * step, max_dist)
        if gap(distance) <= 0.0:
            hi = distance
            break
        lo = distance
        if distance >= max_dist:
            break

    if hi is None:
        return {"hit": False, "t": max_dist, "point": end_point(max_dist)}

    # Bisection: ``lo`` is above the surface, ``hi`` is below it.
    while hi - lo > 1e-9:
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    distance = 0.5 * (lo + hi)
    return {"hit": True, "t": distance, "point": end_point(distance)}


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
    layout = _layout(field)
    point = _vector(center, 3, "center")
    radius = _finite(radius, "radius")
    if radius < 0.0:
        raise TerrainError("radius must not be negative")
    return _resolve_sphere(layout, point, radius)


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
    layout = _layout(field)
    ax, az = _vector(a, 2, "a")
    bx, bz = _vector(b, 2, "b")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TerrainError("steps must be an integer >= 1")
    if steps < 1:
        raise TerrainError("steps must be an integer >= 1")

    heights: list[float] = []
    for index in range(steps + 1):
        t = index / steps
        heights.append(_sample_height(layout, ax + (bx - ax) * t, az + (bz - az) * t))
    return heights


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
    layout = _layout(field)

    if not isinstance(config, Mapping):
        raise TerrainError("config must be a mapping")
    if not isinstance(bodies, (list, tuple)):
        raise TerrainError("bodies must be a list of body mappings")

    default_radius = config.get("radius", _MISSING)
    resolved: list[dict] = []

    for index, body in enumerate(bodies):
        if not isinstance(body, Mapping):
            return {
                "ok": False,
                "bodies": [],
                "error": f"body {index} is not a mapping",
            }

        try:
            center = _vector(body.get("center", _MISSING), 3, f"body {index} center")
        except TerrainError as exc:
            return {"ok": False, "bodies": [], "error": str(exc)}

        radius_value = body.get("radius", _MISSING)
        if radius_value is _MISSING:
            radius_value = default_radius
        try:
            radius = _finite(radius_value, f"body {index} radius")
        except TerrainError as exc:
            return {"ok": False, "bodies": [], "error": str(exc)}
        if radius <= 0.0:
            return {
                "ok": False,
                "bodies": [],
                "error": f"body {index} radius must be greater than zero",
            }

        # Resolved independently from its own position/radius - no shared
        # sampling cache between bodies.
        result = _resolve_sphere(layout, center, radius)
        resolved.append(
            {
                "id": body.get("id"),
                "radius": radius,
                "center": result["center"],
                "corrected": result["corrected"],
            }
        )

    return {"ok": True, "bodies": resolved, "error": None}
