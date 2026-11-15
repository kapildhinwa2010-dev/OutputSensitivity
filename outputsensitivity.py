# OutputSensitivity - Rank layers by how much they move the model output
# Part of the Kapildhinwa2010-dev Toolkit.

"""OutputSensitivity - layer-wise spectral and output-sensitivity analysis.

Lineage: Reverse.Dean (b2/b4). Given a model dict {name: 2-D matrix}
(layers applied in insertion order), this module computes per-layer
singular spectra, effective ranks and condition numbers, and - the namesake
feature - output_sensitivity(): how much the final output moves when each
layer alone is perturbed with Gaussian noise.

Honest limitation: sensitivity here is a one-shot finite difference under
random noise, not a gradient; it is order-dependent and noisy by nature.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "layer_spectrum",
    "effective_rank",
    "layer_report",
    "analyze_model",
    "find_bottlenecks",
    "forward",
    "output_sensitivity",
]


def layer_spectrum(weights: np.ndarray) -> np.ndarray:
    """Singular values of weights in descending order."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2-D matrix")
    return np.linalg.svd(weights, compute_uv=False)


def effective_rank(weights: np.ndarray, energy: float = 0.99) -> int:
    """Smallest k whose singular values cover energy of total energy."""
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    s = layer_spectrum(weights)
    total = float(s.sum())
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(s) / total
    k = int(np.searchsorted(cumulative, energy, side="left") + 1)
    return int(min(k, s.size))


def layer_report(weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.size == 0:
        raise ValueError("weights must be a non-empty 2-D matrix")
    s = layer_spectrum(weights)
    s_max = float(s[0])
    s_min = float(s[-1])
    return {
        "shape": (int(weights.shape[0]), int(weights.shape[1])),
        "frobenius_norm": float(np.linalg.norm(weights)),
        "spectral_norm": s_max,
        "condition_number": (s_max / s_min) if s_min > 0 else float("inf"),
        "effective_rank": effective_rank(weights),
    }


def analyze_model(layers: dict) -> dict:
    """Report every layer of a model dict {name: 2-D matrix}."""
    if not layers:
        raise ValueError("layers must be non-empty")
    return {name: layer_report(w) for name, w in layers.items()}


def find_bottlenecks(reports: dict, top_k: int = 1) -> list:
    """Layer names with the smallest effective rank, worst first."""
    if not reports:
        return []
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ranked = sorted(reports.items(), key=lambda kv: (kv[1]["effective_rank"], kv[0]))
    return [name for name, _ in ranked[:top_k]]


def forward(layers: dict, x: np.ndarray) -> np.ndarray:
    """Apply layers in insertion order: x -> W1 -> W2 -> ... """
    if not layers:
        raise ValueError("layers must be non-empty")
    out = np.asarray(x, dtype=np.float64).ravel()
    for name, w in layers.items():
        w = np.asarray(w, dtype=np.float64)
        if w.ndim != 2:
            raise ValueError("layer {} must be a 2-D matrix".format(name))
        if out.size != w.shape[0]:
            raise ValueError(
                "layer {} expects input dim {}, got {}".format(name, w.shape[0], out.size)
            )
        out = w @ out
    return out


def output_sensitivity(layers: dict, x: np.ndarray, noise_std: float = 0.01,
                        repeats: int = 5, rng: np.random.Generator = None) -> list:
    """Rank layers by how much perturbing them moves the final output.

    For each layer and each of repeats trials, Gaussian noise of standard
    deviation noise_std is added to that layer alone and the relative
    change of the final output is measured. Returns dicts with layer,
    mean_relative_change and baseline_output_norm, most sensitive first.
    This is a finite-difference estimate, not a gradient.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    if noise_std < 0 or repeats < 1:
        raise ValueError("noise_std must be >= 0 and repeats >= 1")
    rng = rng or np.random.default_rng(0)
    base = forward(layers, x)
    base_norm = float(np.linalg.norm(base))
    rows = []
    for name, w in layers.items():
        w = np.asarray(w, dtype=np.float64)
        total = 0.0
        for _ in range(repeats):
            perturbed = dict(layers)
            perturbed[name] = w + rng.normal(0.0, noise_std, size=w.shape)
            out = forward(perturbed, x)
            total += float(np.linalg.norm(out - base)) / base_norm if base_norm else 0.0
        rows.append({
            "layer": name,
            "mean_relative_change": total / repeats,
            "baseline_output_norm": base_norm,
        })
    rows.sort(key=lambda row: row["mean_relative_change"], reverse=True)
    return rows
