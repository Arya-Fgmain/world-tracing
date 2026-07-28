#!/usr/bin/env python3
"""Compare a raw WT XYZ NPZ with an ordered, static author WTPC reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", required=True, type=Path)
    parser.add_argument("--author-wtpc", required=True, type=Path)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def load_wtpc_points(path: Path) -> np.ndarray:
    """Read positions from a static WTPC without importing the model package."""
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:4] != b"WTPC" or raw[5] != 0:
        raise ValueError(f"Expected a static WTPC file: {path}")
    offset = 12 + (24 if raw[8] == 1 else 0)
    if len(raw) < offset + 4:
        raise ValueError(f"Truncated WTPC point count: {path}")
    count = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    positions_end = offset + count * 3 * np.dtype("<f4").itemsize
    if len(raw) < positions_end:
        raise ValueError(f"Truncated WTPC positions: {path}")
    return np.frombuffer(
        raw, dtype="<f4", count=count * 3, offset=offset
    ).reshape(count, 3).copy()


def fit_similarity(source: np.ndarray, target: np.ndarray):
    """Fit ``target ~= scale * (source - mean) @ rotation + target_mean``."""
    source64 = np.asarray(source, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    source_mean = source64.mean(axis=0)
    target_mean = target64.mean(axis=0)
    source_centered = source64 - source_mean
    target_centered = target64 - target_mean
    covariance = source_centered.T @ target_centered / len(source64)
    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = (u * signs) @ vt
    variance = np.square(source_centered).sum() / len(source64)
    scale = float(np.dot(singular_values, signs) / variance)
    translation = target_mean - scale * (source_mean @ rotation)
    return scale, rotation, translation


def error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def optional_npz_scalar(data, name: str):
    if name not in data:
        return None
    value = data[name]
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def main() -> None:
    args = parse_args()
    with np.load(args.xyz) as data:
        xyz = data["xyz"].astype(np.float64)
        mask = data["mask"].astype(bool)
        run_metadata = {
            name: optional_npz_scalar(data, name)
            for name in (
                "seed",
                "config",
                "checkpoint_resolved",
                "checkpoint_sha256",
                "autocast_dtype",
                "matmul_allow_tf32",
                "cudnn_allow_tf32",
                "num_steps",
                "trellis_seed",
            )
        }

    if xyz.ndim != 4 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected XYZ shaped [layers, H, W, 3], got {xyz.shape}")
    if mask.shape != xyz.shape[:-1]:
        raise ValueError(f"Mask shape {mask.shape} does not match XYZ {xyz.shape}")

    local_layers = [xyz[layer][mask[layer]] for layer in range(len(xyz))]
    counts = [len(points) for points in local_layers]
    author = load_wtpc_points(args.author_wtpc).astype(np.float64)
    if sum(counts) != len(author):
        raise ValueError(
            "Ordered comparison requires matching point counts: "
            f"local={sum(counts)}, author={len(author)}, layers={counts}"
        )

    offsets = np.cumsum([0, *counts])
    author_layers = [
        author[offsets[layer] : offsets[layer + 1]]
        for layer in range(len(counts))
    ]
    scale, rotation, translation = fit_similarity(
        local_layers[0], author_layers[0]
    )
    aligned_layers = [
        scale * (points @ rotation) + translation for points in local_layers
    ]
    per_layer = []
    for layer, (aligned, reference) in enumerate(
        zip(aligned_layers, author_layers, strict=True)
    ):
        errors = np.linalg.norm(aligned - reference, axis=1)
        per_layer.append(
            {
                "layer": layer + 1,
                "point_count": len(errors),
                **error_summary(errors),
            }
        )

    hidden_p90_mean = float(
        np.mean([metrics["p90"] for metrics in per_layer[1:]])
    )
    metrics = {
        "xyz": str(args.xyz),
        "author_wtpc": str(args.author_wtpc),
        "run": run_metadata,
        "point_counts_by_layer": counts,
        "alignment": {
            "fit_layer": 1,
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "per_layer_correspondence_error": per_layer,
        "hidden_layer_p90_mean": hidden_p90_mean,
    }

    masks_identical = all(np.array_equal(mask[0], item) for item in mask[1:])
    equal_counts = len(set(counts)) == 1
    if masks_identical and equal_counts:
        local_stack = np.stack(aligned_layers)
        author_stack = np.stack(author_layers)
        local_span = (
            local_stack[..., 2].max(axis=0)
            - local_stack[..., 2].min(axis=0)
        )
        author_span = (
            author_stack[..., 2].max(axis=0) - author_stack[..., 2].min(axis=0)
        )
        metrics["per_ray_depth_span"] = {
            "local_median": float(np.median(local_span)),
            "author_median": float(np.median(author_span)),
            "pearson_correlation": float(
                np.corrcoef(local_span, author_span)[0, 1]
            ),
        }

    print(
        "Run: "
        f"WT seed={run_metadata['seed']}, "
        f"TRELLIS seed={run_metadata['trellis_seed']}, "
        f"autocast={run_metadata['autocast_dtype']}, "
        f"steps={run_metadata['num_steps']}, "
        f"TF32(matmul={run_metadata['matmul_allow_tf32']}, "
        f"cudnn={run_metadata['cudnn_allow_tf32']})"
    )
    print(f"Visible-layer alignment scale: {scale:.8f}")
    for item in per_layer:
        print(
            f"Layer {item['layer']}: median={item['median']:.8f}, "
            f"p90={item['p90']:.8f}, p95={item['p95']:.8f}, "
            f"rmse={item['rmse']:.8f}"
        )
    print(f"Hidden-layer mean p90: {hidden_p90_mean:.8f}")
    if "per_ray_depth_span" in metrics:
        span = metrics["per_ray_depth_span"]
        print(
            "Per-ray depth span: "
            f"local median={span['local_median']:.8f}, "
            f"author median={span['author_median']:.8f}, "
            f"correlation={span['pearson_correlation']:.8f}"
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Wrote JSON metrics: {args.json}")


if __name__ == "__main__":
    main()
