"""Decode an authors-released WTPC through the local TRELLIS.2 bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wt.data import load_rgba_image
from wt.textured_mesh import (
    compute_canonical_transform,
    inject_coords_into_trellis2,
    load_trellis2_pipeline,
    point_cloud_fill,
    save_mesh_glb,
)
from wt.wtpc import load_static_wtpc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wtpc", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--voxel-npz", type=Path)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--pipeline-type", default="512", choices=("512", "1024"))
    parser.add_argument("--ss-res", type=int, default=32)
    parser.add_argument("--close-iters", type=int, default=1)
    parser.add_argument(
        "--fill-holes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--canonical-axis-map",
        default="cam2canonical_zup_upstream",
        choices=(
            "cam2canonical_zup",
            "cam2canonical_zup_upstream",
            "identity",
            "y_flip",
        ),
    )
    parser.add_argument(
        "--trellis-preprocess-image",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--trellis2-model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--trellis2-path", type=Path)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    args = parser.parse_args()

    payload = load_static_wtpc(args.wtpc)
    points = payload.points
    print(
        f"[wtpc] version={payload.version}; points={len(points):,}; "
        f"colors={'yes' if payload.colors is not None else 'no'}"
    )
    print(
        f"[wtpc] camera bounds: min={points.min(axis=0).tolist()}, "
        f"max={points.max(axis=0).tolist()}"
    )

    transform = compute_canonical_transform(
        points,
        half_target=0.45,
        axis_map=args.canonical_axis_map,
    )
    diagnostics: dict[str, np.ndarray] = {}
    coords, n_voxels = point_cloud_fill(
        points,
        transform.apply,
        res=args.ss_res,
        close_iters=args.close_iters,
        fill_holes=args.fill_holes,
        seed=args.seed,
        diagnostics=diagnostics,
    )
    counts = {name: int(grid.sum()) for name, grid in diagnostics.items()}
    print(f"[wtpc] occupied voxels={counts}; sent to TRELLIS.2={n_voxels:,}")
    if len(coords) == 0:
        raise RuntimeError("WTPC voxelization produced no active coordinates")

    if args.voxel_npz is not None:
        args.voxel_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.voxel_npz,
            **diagnostics,
            source_points=points,
            resolution=np.int64(args.ss_res),
            close_iters=np.int64(args.close_iters),
            fill_holes=np.bool_(args.fill_holes),
            canonical_centroid=transform.centroid,
            canonical_scale=np.float64(transform.scale),
            canonical_axis_map=np.array(transform.axis_map),
        )
        print(f"[wtpc] wrote voxel diagnostics: {args.voxel_npz}")

    rgba = load_rgba_image(args.image, auto_alpha=False)
    pil_rgba = Image.fromarray(rgba, mode="RGBA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[wtpc] loading TRELLIS.2 pipeline on {device} ...")
    pipeline = load_trellis2_pipeline(
        model_id=args.trellis2_model,
        device=device,
        trellis2_path=args.trellis2_path,
    )
    print(
        f"[wtpc] decoding TRELLIS.2 ({args.pipeline_type}, seed={args.seed}, "
        f"preprocess_image={args.trellis_preprocess_image}) ..."
    )
    meshes = inject_coords_into_trellis2(
        pipeline,
        pil_rgba,
        coords,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        preprocess_image=args.trellis_preprocess_image,
    )
    output = save_mesh_glb(
        args.out,
        meshes[0],
        texture_size=args.texture_size,
        decimation_target=args.decimation_target,
    )
    print(f"[wtpc] wrote {output}")


if __name__ == "__main__":
    main()
