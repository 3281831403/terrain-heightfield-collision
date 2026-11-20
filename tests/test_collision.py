"""Behavioural tests for the terrain height-field collision helpers.

Every test talks to the public API of ``field.collision`` only.  Heights are in
meters, a grid sample ``heights[iz][ix]`` sits at
``(origin[0] + ix * spacing, origin[1] + iz * spacing)``, and the numeric
tolerance is the documented ``1e-6``.
"""

import math

import pytest

from field.collision import (
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

TOL = 1e-6
SQRT1_2 = 1.0 / math.sqrt(2.0)


def _grid(nx, nz, fn, origin=(0.0, 0.0), spacing=1.0):
    """Build a height field of ``nx * nz`` samples with ``h = fn(ix, iz)``."""
    heights = [[float(fn(ix, iz)) for ix in range(nx)] for iz in range(nz)]
    return {"origin": [float(origin[0]), float(origin[1])], "spacing": float(spacing), "heights": heights}


# ---------------------------------------------------------------------------
# one test per seeded defect
# ---------------------------------------------------------------------------


def test_bilinear_interpolation_matches_linear_plane():
    """Defect 1: rounding to the nearest cell centre turns a slope into stairs."""
    field = _grid(3, 3, lambda ix, iz: ix + 2 * iz)  # h = x + 2z over [0, 2] m
    assert sample_height(field, 0.25, 0.5) == pytest.approx(1.25, abs=TOL)
    assert sample_height(field, 1.5, 0.75) == pytest.approx(3.0, abs=TOL)
    assert sample_height(field, 0.125, 1.75) == pytest.approx(3.625, abs=TOL)


def test_sample_height_clamps_outside_grid():
    """Defect 2: no clamping, so out-of-range positions index out of bounds."""
    field = _grid(3, 3, lambda ix, iz: ix + 2 * iz)
    assert sample_height(field, -3.0, -4.0) == pytest.approx(0.0, abs=TOL)
    assert sample_height(field, 9.0, 9.0) == pytest.approx(6.0, abs=TOL)
    assert sample_height(field, -0.5, 0.5) == pytest.approx(1.0, abs=TOL)
    assert sample_height(field, 1.5, 9.0) == pytest.approx(5.5, abs=TOL)


def test_sample_normal_is_unit_length_on_slope():
    """Defect 3: the raw difference vector is neither normalized nor sign-correct."""
    field = _grid(2, 2, lambda ix, iz: ix)  # h = x, a 45 degree ramp
    n = sample_normal(field, 0.5, 0.5)
    assert len(n) == 3
    assert math.sqrt(sum(c * c for c in n)) == pytest.approx(1.0, abs=TOL)
    assert n[0] == pytest.approx(-SQRT1_2, abs=TOL)
    assert n[1] == pytest.approx(SQRT1_2, abs=TOL)
    assert n[2] == pytest.approx(0.0, abs=TOL)


def test_slope_deg_uses_grid_spacing():
    """Defect 4: slope from raw height difference ignores the grid spacing."""
    steep = _grid(4, 4, lambda ix, iz: ix, spacing=0.5)  # 1 m rise per 0.5 m -> 2.0
    gentle = _grid(2, 2, lambda ix, iz: ix, spacing=2.0)  # 1 m rise per 2 m -> 0.5
    assert slope_deg(steep, 0.5, 0.5) == pytest.approx(math.degrees(math.atan(2.0)), abs=TOL)
    assert slope_deg(gentle, 1.0, 1.0) == pytest.approx(math.degrees(math.atan(0.5)), abs=TOL)


def test_raycast_refines_hit_point():
    """Defect 5: a pure march stops half a cell early and misses the surface."""
    field = _grid(3, 3, lambda ix, iz: ix)  # h = x
    result = raycast(field, [-1.0, 2.0, 0.5], [1.0, -1.0, 0.0], 10.0)
    assert result["hit"] is True
    assert result["t"] == pytest.approx(3.0 / math.sqrt(2.0), abs=1e-6)
    px, py, pz = result["point"]
    assert px == pytest.approx(0.5, abs=1e-6)
    assert py == pytest.approx(0.5, abs=1e-6)
    assert pz == pytest.approx(0.5, abs=1e-6)
    assert py == pytest.approx(sample_height(field, px, pz), abs=1e-6)


def test_resolve_sphere_pushes_along_surface_normal():
    """Defect 6: a vertical-only lift leaves the sphere buried on a slope."""
    field = _grid(3, 3, lambda ix, iz: ix)  # h = x, the plane y = x
    center = [1.0, 0.2, 0.5]
    radius = 0.5
    result = resolve_sphere(field, center, radius)
    assert result["corrected"] is True
    assert result["penetration"] > 0.0
    new_center = result["center"]
    assert len(new_center) == 3

    normal = [-SQRT1_2, SQRT1_2, 0.0]
    distance = sum(c * n for c, n in zip(normal, new_center))
    assert distance == pytest.approx(radius, abs=1e-6)

    delta = [new - old for new, old in zip(new_center, center)]
    assert delta[0] < 0.0
    assert delta[1] > 0.0
    assert delta[2] == pytest.approx(0.0, abs=TOL)
    cross = [
        delta[1] * normal[2] - delta[2] * normal[1],
        delta[2] * normal[0] - delta[0] * normal[2],
        delta[0] * normal[1] - delta[1] * normal[0],
    ]
    for component in cross:
        assert component == pytest.approx(0.0, abs=1e-6)


def test_height_over_segment_samples_uniformly_by_arc_length():
    """Defect 7: stepping by grid or surface length oversamples steep ground."""
    field = _grid(3, 3, lambda ix, iz: 2.0 * ix)  # h = 2x, steep and uneven
    heights = height_over_segment(field, [0.0, 0.0], [2.0, 0.0], 8)
    assert len(heights) == 9
    step = 2.0 / 8.0
    for index, got in enumerate(heights):
        assert got == pytest.approx(2.0 * index * step, abs=TOL)


def test_validate_field_detects_ragged_grid():
    """Defect 8: a non-rectangular grid is accepted and later indexes out of bounds."""
    ragged = {"origin": [0.0, 0.0], "spacing": 1.0, "heights": [[0.0, 1.0, 2.0], [0.0, 1.0]]}
    problems = validate_field(ragged)
    assert isinstance(problems, list)
    assert problems

    empty = {"origin": [0.0, 0.0], "spacing": 1.0, "heights": []}
    assert validate_field(empty)

    assert validate_field(_grid(2, 2, lambda ix, iz: ix)) == []


def test_collide_batch_does_not_share_state_between_bodies():
    """Defect 9: one shared sample cache lets neighbouring bodies overwrite each other."""
    field = _grid(3, 3, lambda ix, iz: ix)  # h = x, 1 m cells
    left_center = [0.2, 0.35, 0.5]
    right_center = [0.7, 0.35, 0.5]
    bodies = [
        {"id": "left", "center": list(left_center), "radius": 0.2},
        {"id": "right", "center": list(right_center), "radius": 0.2},
    ]

    result = collide_batch(field, bodies, {})
    assert result["ok"] is True
    assert result["error"] is None
    assert len(result["bodies"]) == 2

    expected_left = resolve_sphere(field, left_center, 0.2)["center"]
    expected_right = resolve_sphere(field, right_center, 0.2)["center"]
    assert result["bodies"][0]["center"] == pytest.approx(expected_left, abs=1e-6)
    assert result["bodies"][1]["center"] == pytest.approx(expected_right, abs=1e-6)
    assert expected_left[0] != pytest.approx(expected_right[0], abs=1e-3)


def test_non_positive_spacing_raises_terrain_error():
    """Defect 10: spacing <= 0 is not validated, so interpolation divides by zero."""
    flat_corners = [[0.0, 1.0], [1.0, 2.0]]
    zero = {"origin": [0.0, 0.0], "spacing": 0.0, "heights": flat_corners}
    with pytest.raises(TerrainError):
        sample_height(zero, 0.5, 0.5)
    assert validate_field(zero)

    negative = {"origin": [0.0, 0.0], "spacing": -1.0, "heights": flat_corners}
    assert validate_field(negative)
    with pytest.raises(TerrainError):
        sample_normal(negative, 0.0, 0.0)


# ---------------------------------------------------------------------------
# invariant tests
# ---------------------------------------------------------------------------


def test_flat_field_has_constant_height():
    """A flat field must sample to the same height everywhere."""
    field = _grid(4, 4, lambda ix, iz: 7.0, spacing=0.5)
    for x, z in [(0.0, 0.0), (0.25, 0.75), (1.5, 1.5), (-5.0, 3.0), (40.0, -40.0)]:
        assert sample_height(field, x, z) == pytest.approx(7.0, abs=TOL)


def test_normal_points_up_on_flat_field():
    """The flat-ground normal is so that walkability math has a stable baseline."""
    field = _grid(3, 3, lambda ix, iz: 2.0)
    n = sample_normal(field, 0.5, 0.5)
    assert n[0] == pytest.approx(0.0, abs=TOL)
    assert n[1] == pytest.approx(1.0, abs=TOL)
    assert n[2] == pytest.approx(0.0, abs=TOL)
    assert slope_deg(field, 0.5, 0.5) == pytest.approx(0.0, abs=TOL)


def test_raycast_reports_miss():
    """A ray that never reaches the terrain reports the miss and the ray end."""
    field = _grid(3, 3, lambda ix, iz: ix)
    result = raycast(field, [0.5, 5.0, 0.5], [0.0, 1.0, 0.0], 4.0)
    assert result["hit"] is False
    assert result["t"] == pytest.approx(4.0, abs=TOL)
    assert result["point"] == pytest.approx([0.5, 9.0, 0.5], abs=TOL)


def test_collide_batch_preserves_input_order():
    """A fully successful batch returns one entry per body in input order."""
    field = _grid(3, 3, lambda ix, iz: ix)
    bodies = [
        {"id": "c", "center": [1.0, 1.5, 1.0], "radius": 0.2},
        {"id": "a", "center": [0.5, 1.5, 0.5], "radius": 0.2},
        {"id": "b", "center": [1.5, 1.5, 1.5], "radius": 0.2},
    ]
    result = collide_batch(field, bodies, {"radius": 0.2})
    assert result["ok"] is True
    assert result["error"] is None
    assert [body["id"] for body in result["bodies"]] == ["c", "a", "b"]
    assert [body["radius"] for body in result["bodies"]] == pytest.approx([0.2, 0.2, 0.2], abs=TOL)


def test_heightfield_pipeline_stays_consistent():
    """Combined check: validation, sampling, normals, slope, glue and raycast agree."""
    field = _grid(3, 3, lambda ix, iz: ix)  # h = x, 1 m cells over [0, 2] m

    assert validate_field(field) == []
    assert sample_height(field, 1.25, 0.0) == pytest.approx(1.25, abs=TOL)

    normal = sample_normal(field, 1.0, 1.0)
    assert slope_deg(field, 1.0, 1.0) == pytest.approx(45.0, abs=TOL)
    assert normal[1] == pytest.approx(math.cos(math.radians(45.0)), abs=TOL)
    assert normal[2] == pytest.approx(0.0, abs=TOL)

    assert height_over_segment(field, [0.0, 1.0], [2.0, 1.0], 2) == pytest.approx([0.0, 1.0, 2.0], abs=TOL)

    hit = raycast(field, [-1.0, 2.0, 0.5], [1.0, -1.0, 0.0], 10.0)
    assert hit["hit"] is True
    hx, hy, hz = hit["point"]
    assert 0.0 <= hx <= 2.0
    assert hy == pytest.approx(sample_height(field, hx, hz), abs=1e-6)
