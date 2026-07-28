"""Reader for static World Tracing point-cloud (``.wtpc``) files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class StaticWTPC:
    """Decoded static WTPC payload."""

    points: np.ndarray
    colors: np.ndarray | None
    header_data: np.ndarray | None
    version: int


def load_static_wtpc(path: str | Path) -> StaticWTPC:
    """Load a static WTPC payload; reject animated or truncated files."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:4] != b"WTPC":
        raise ValueError(f"Not a WTPC file: {path}")
    if raw[5] != 0:
        raise ValueError(f"Expected a static WTPC file: {path}")

    version = int(raw[4])
    offset = 12
    header_data = None
    if raw[8] == 1:
        if len(raw) < offset + 24:
            raise ValueError(f"Truncated WTPC extended header: {path}")
        header_data = np.frombuffer(
            raw, dtype="<f4", count=6, offset=offset
        ).copy()
        offset += 24

    if len(raw) < offset + 4:
        raise ValueError(f"Truncated WTPC point count: {path}")
    count = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    positions_end = offset + count * 3 * np.dtype("<f4").itemsize
    if len(raw) < positions_end:
        raise ValueError(
            f"Truncated WTPC positions: expected {count} points in {path}"
        )
    points = np.frombuffer(
        raw, dtype="<f4", count=count * 3, offset=offset
    ).reshape(count, 3).copy()

    remaining = len(raw) - positions_end
    colors = None
    if remaining == count * 3:
        colors = np.frombuffer(
            raw, dtype=np.uint8, count=count * 3, offset=positions_end
        ).reshape(count, 3).copy()
    elif remaining != 0:
        raise ValueError(
            f"Unsupported WTPC trailing payload: {remaining} bytes in {path}"
        )

    return StaticWTPC(
        points=points,
        colors=colors,
        header_data=header_data,
        version=version,
    )


__all__ = ["StaticWTPC", "load_static_wtpc"]
