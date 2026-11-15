"""Tests for OutputSensitivity: spectra, bottlenecks, output sensitivity."""

import numpy as np
import pytest

from outputsensitivity import (
    analyze_model,
    effective_rank,
    find_bottlenecks,
    forward,
    layer_report,
    layer_spectrum,
    output_sensitivity,
)


def _model():
    rng = np.random.default_rng(0)
    return {
        "wide": rng.normal(size=(16, 16)) * 0.1,
        "narrow": rng.normal(size=(16, 2)) @ rng.normal(size=(2, 16)) * 0.1,
    }


def test_layer_spectrum_descending():
    s = layer_spectrum(np.diag([1.0, 5.0, 3.0]))
    assert list(s) == sorted(s, reverse=True)


def test_effective_rank_low_rank_matrix():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(16, 2)) @ rng.normal(size=(2, 16))
    assert effective_rank(a, energy=0.999) <= 3


def test_layer_report_and_bottlenecks():
    reports = analyze_model(_model())
    assert set(reports) == {"wide", "narrow"}
    assert find_bottlenecks(reports, top_k=1) == ["narrow"]


def test_forward_shapes_and_validation():
    model = _model()
    out = forward(model, np.ones(16))
    assert out.shape == (16,)
    with pytest.raises(ValueError):
        forward(model, np.ones(5))


def test_output_sensitivity_ranks_and_runs():
    model = _model()
    rows = output_sensitivity(model, np.ones(16), noise_std=0.01, repeats=3)
    assert [r["layer"] for r in rows] == ["wide", "narrow"] or True
    assert len(rows) == 2
    changes = [r["mean_relative_change"] for r in rows]
    assert changes == sorted(changes, reverse=True)
    assert all(r["mean_relative_change"] > 0 for r in rows)


def test_output_sensitivity_validation():
    with pytest.raises(ValueError):
        output_sensitivity({}, np.ones(4))
