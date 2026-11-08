#!/usr/bin/env python3
"""gem.py — procesa vídeos y deja datos de movimiento en NPZ + BVH.

Flujo
-----
1. Prepara el motor de captura y sus dependencias (una vez).
2. Lee la celda 2: cualquier cantidad de rutas de vídeo entre comillas se
   procesa en lote.  Vale una ruta, una tupla, una lista o varias cadenas
   seguidas; lo importante es que cada ruta esté entre comillas.
3. Por cada vídeo deja exactamente dos ficheros en `GEM_OUTPUT_DIR`
   (`/kaggle/working/output` por defecto):
       <nombre>.npz   datos de movimiento
       <nombre>.bvh   esqueleto animado para inspección

No escribe intermedios, auditorías ni subcarpetas.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tokenize
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
from IPython import get_ipython


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
WORK = Path(os.environ.get("GEM_WORK_DIR", "/kaggle/working"))
ENGINE_ROOT = WORK / "GEM-X"
OUTPUT_DIR = Path(os.environ.get("GEM_OUTPUT_DIR", WORK / "output"))
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

MAX_W = int(os.environ.get("GEM_MAX_W", "1280"))
MAX_H = int(os.environ.get("GEM_MAX_H", "720"))
MAX_FPS = float(os.environ.get("GEM_MAX_FPS", "30"))
CRF = int(os.environ.get("GEM_CRF", "23"))
PRESET = os.environ.get("GEM_PRESET", "veryfast")
DEVICE = os.environ.get("GEM_DEVICE", "cuda")
CAMERA = os.environ.get("GEM_CAMERA", "static")

ONNX_VERSION = "1.22.0"
PIP_EXTRA = ("nvidia-cufft-cu12", "nvidia-curand-cu12",
             "nvidia-cuda-nvrtc-cu12")

WEIGHTS = ENGINE_ROOT / "weights"
SOURCE_ASSETS = ENGINE_ROOT / "assets" / "soma"

HF_FILES = (
    # (repo_id, revision, filename, destino)
    ("nvidia/GEM-X", None, "motion_native77_e774_s292175.ckpt", WEIGHTS),
    ("nvidia/GEM-X", None, "motion_stats_raw.npz", WEIGHTS),
    ("nvidia/GEM-X", None, "body_video_native77.yaml", WEIGHTS),
    ("nvidia/GEM-X", None, "body_s240000.ckpt", WEIGHTS),
    ("nvidia/GEM-X", None, "vitpose.pth", WEIGHTS),
    ("nvidia/SOMA-X", "v0.3.3", "SOMAHand.npz", SOURCE_ASSETS),
    ("nvidia/SOMA-X", "v0.3.3", "MHR/mhr_model_lod1.pt", SOURCE_ASSETS),
    ("nvidia/SOMA-X", "v0.3.3", "MHR/base_body_lod1.obj", SOURCE_ASSETS),
    ("nvidia/SOMA-X", "v0.3.3", "MHR/SOMA_wrap_lod1.obj", SOURCE_ASSETS),
    ("nvidia/SOMA-X", "v0.3.3", "SOMA_neutral.npz", SOURCE_ASSETS),
    ("nvidia/SOMA-X", "v0.3.3", "SOMA_template_rig.usda", SOURCE_ASSETS),
    ("nvidia/SOMA-X", "v0.3.3", "SOMA_procedural_transforms.json",
     SOURCE_ASSETS),
    ("nvidia/SOMA-X", "v0.3.3", "correctives_model.pt", SOURCE_ASSETS),
)


# --------------------------------------------------------------------------- #
# Activos embebidos
# --------------------------------------------------------------------------- #
_SOURCE_ASSET_B64 = (
    "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"
)

_REST_B64 = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAM2spIO1/sD9cmEwVjEq6v8gBu5o8ZYW8UMWNW8zPvT8+"
    "Zfz7jAvBv+dHwylz84+8UMWNW8zPvT9GFjCBW3fQv1gceLXcmZk/1rLG6qBcvj82iUFg5dDbvzPE"
    "lbN3Rqu/ZQOXZHBWvj8Y529CIYLnvzzElbN3Rqu/ZQOXZHBWvj/O9ugN9xHovzzElbN3Rqu/ZQOX"
    "ZHBWvj/te9RfrzDpv7jBpWPOM7Y/M2spIO1/sL9cmEwVjEq6v1JaRuo9lWO8UMWNW8zPvb8+Zfz7"
    "jAvBv80UrHE2HVG8UMWNW8zPvb9GFjCBW3fQv2gceLXcmZk/2LLG6qBcvr82iUFg5dDbvyzElbN3"
    "Rqu/aAOXZHBWvr8Y529CIYLnvzbElbN3Rqu/aAOXZHBWvr/O9ugN9xHovzbElbN3Rqu/aAOXZHBW"
    "vr/te9RfrzDpv7zBpWPOM7Y/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9goL7gc8MLzDSQwCK4em"
    "P+oKC+4HPHC/9goL7gc8MLzDSQwCK4emP+oKC+4HPHC/8rVnlgSouT/PHmgFhqzSP/N9E+7wMt6+"
    "b5nTZTGxwT+KLT2a6snRPwezE+7wMt6+r0LKT6p9wj84HQHcLF7GPxcjFO7wMt6+r0LKT6p9wj/X"
    "9UTXhR+4P9vfRQSIJ5A/mNukP4e7wj9IM0+uKZC1P5ERq5p7o70/mNukP4e7wj9IM0+uKZC1Pz9H"
    "9efsrsM/mNukP4e7wj9IM0+uKZC1P1YgxN9Aksk/mNukP4e7wj9IM0+uKZC1P5F2SNaGL9M/1m67"
    "0Fynub/PHmgFhqzSPyexEu7wMt6+4XX9gt2wwb+KLT2a6snRP2eYEu7wMt6+IR/0bFZ9wr84HQHc"
    "LF7GP7D7Eu7wMt6+IR/0bFZ9wr/X9UTXhR+4P+7fRQSIJ5A/CLjOXDO7wr9IM0+uKZC1P5URq5p7"
    "o70/CLjOXDO7wr9IM0+uKZC1P0FH9efsrsM/CLjOXDO7wr9IM0+uKZC1P1ggxN9Aksk/CLjOXDO7"
    "wr9IM0+uKZC1P5N2SNaGL9M/"
)


# --------------------------------------------------------------------------- #
# Estructura de salida (tablas; el resto se deriva)
# --------------------------------------------------------------------------- #
NAMES = [
    "pelvis_skel",
    "left_hip_pitch_skel", "left_hip_roll_skel", "left_hip_yaw_skel",
    "left_knee_skel", "left_ankle_pitch_skel", "left_ankle_roll_skel",
    "left_toe_base",
    "right_hip_pitch_skel", "right_hip_roll_skel", "right_hip_yaw_skel",
    "right_knee_skel", "right_ankle_pitch_skel", "right_ankle_roll_skel",
    "right_toe_base",
    "waist_yaw_skel", "waist_roll_skel", "waist_pitch_skel",
    "left_shoulder_pitch_skel", "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel", "left_elbow_skel",
    "left_wrist_roll_skel", "left_wrist_pitch_skel", "left_wrist_yaw_skel",
    "left_hand_roll_skel",
    "right_shoulder_pitch_skel", "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel", "right_elbow_skel",
    "right_wrist_roll_skel", "right_wrist_pitch_skel", "right_wrist_yaw_skel",
    "right_hand_roll_skel",
]

PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13,
           0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27, 28,
           29, 30, 31, 32]

FEET = (6, 7, 13, 14)
DEAD_LEAVES = (7, 14, 25, 33)
R_HIP = 8
L_HIP = 1

JOINTS = len(NAMES)
FPS = 30.0


def _layout(joins: int, feet: int) -> dict:
    """Reparte el vector en bloques contiguos a partir de sus tamaños."""
    sizes = (
        ("global_root", 5),
        ("root_local", 4),
        ("ric_data", (joins - 1) * 3),
        ("global_rot_data", joins * 6),
        ("local_vel", joins * 3),
        ("foot_contacts", feet),
    )
    spans, start = {}, 0
    for block, size in sizes:
        spans[block] = (start, start + size)
        start += size
    return spans


LAYOUT = _layout(JOINTS, len(FEET))
DIM = LAYOUT["foot_contacts"][1]

SMOOTH_SIGMA = 1.0
H_ON = 0.06
H_OFF = 0.10
V_ON = 0.30
V_OFF = 0.80


# --------------------------------------------------------------------------- #
# Utilidades numéricas
# --------------------------------------------------------------------------- #
def wrap(a):
    """Envuelve un ángulo (o array) a (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def rot6d_from_mats(mats: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): las dos primeras columnas aplanadas."""
    return np.concatenate([mats[..., :, 0], mats[..., :, 1]], axis=-1)


def cont6d_to_matrix(x: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) por Gram-Schmidt."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def heading_from_pose(pos: np.ndarray, r_hip: int = R_HIP,
                      l_hip: int = L_HIP) -> np.ndarray:
    """Heading (yaw) a partir de la línea entre caderas, en XZ."""
    skel2d = pos.copy()
    skel2d[..., 1] = 0.0
    across = skel2d[:, r_hip, :] - skel2d[:, l_hip, :]
    across = across / (np.linalg.norm(across, axis=-1, keepdims=True) + 1e-6)
    fwd = np.cross(np.array([0.0, 1.0, 0.0]), across)
    fwd = fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-6)
    return np.arctan2(fwd[..., 0], fwd[..., 2])


def load_rest() -> np.ndarray:
    """Posiciones de reposo del esqueleto de salida: (34, 3)."""
    raw = base64.b64decode(_REST_B64)
    return np.frombuffer(raw, dtype="<f8").reshape(JOINTS, 3).copy()


def fk(rest: np.ndarray, parents, root_pos: np.ndarray,
       rot: np.ndarray) -> np.ndarray:
    """Cinemática directa vectorizada."""
    rest = np.asarray(rest, dtype=np.float64)
    parents = np.asarray(parents)
    rot = np.asarray(rot, dtype=np.float64)
    T, J = rot.shape[0], rot.shape[1]
    out = np.zeros((T, J, 3), dtype=np.float64)
    root = int(np.where(parents < 0)[0][0])
    out[:, root] = root_pos
    for j in range(J):
        p = int(parents[j])
        if p < 0:
            continue
        off = rest[j] - rest[p]
        out[:, j] = out[:, p] + np.einsum("tij,j->ti", rot[:, p], off)
    return out


def bone_cv(pos: np.ndarray, parents=None) -> float:
    """CV máximo de las longitudes de hueso (ignora longitud cero)."""
    parents = np.asarray(PARENTS if parents is None else parents)
    worst = 0.0
    for j, p in enumerate(parents):
        if p < 0:
            continue
        lens = np.linalg.norm(pos[:, j] - pos[:, p], axis=-1)
        m = float(lens.mean())
        if m < 1e-6:
            continue
        worst = max(worst, float(lens.std() / m))
    return worst


def canonicalize_pose(pos: np.ndarray, rot: np.ndarray):
    """Quita traslación y giro iniciales; devuelve pose y heading."""
    heading = heading_from_pose(pos)
    init = float(heading[0])
    root_xz = pos[0, 0, [0, 2]].copy()
    pt = pos.copy()
    pt[..., 0] -= root_xz[0]
    pt[..., 2] -= root_xz[1]
    c, s = np.cos(-init), np.sin(-init)
    ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    pc = pt @ ry.T
    rc = np.einsum("ab,tjbc->tjac", ry, rot)
    hc = wrap(heading - init)
    return pc, rc, hc


def encode_pose(pos_c: np.ndarray, rot_c: np.ndarray,
                heading_c: np.ndarray, contacts: np.ndarray,
                fps: float = FPS) -> np.ndarray:
    """Pose canónica -> vector de rasgos."""
    pos_c = np.asarray(pos_c, dtype=np.float64)
    rot_c = np.asarray(rot_c, dtype=np.float64)
    T, J = pos_c.shape[0], pos_c.shape[1]
    if J != JOINTS:
        raise ValueError(f"se esperaban {JOINTS} joints, hay {J}")

    vel = np.zeros_like(pos_c)
    if T > 1:
        vel[:-1] = (pos_c[1:] - pos_c[:-1]) * fps
        vel[-1] = vel[-2]

    ric = pos_c.copy()
    ric[..., 0] -= ric[:, 0:1, 0]
    ric[..., 2] -= ric[:, 0:1, 2]
    ric_data = ric[:, 1:].reshape(T, -1)

    rot6d = rot6d_from_mats(rot_c).reshape(T, -1)

    root = pos_c[:, 0]
    global_root = np.concatenate(
        [root, np.cos(heading_c)[:, None], np.sin(heading_c)[:, None]], axis=-1)

    dhead = np.zeros(T, dtype=np.float64)
    if T > 1:
        dh = wrap(heading_c[1:] - heading_c[:-1]) * fps
        dhead[:-1] = dh
        dhead[-1] = dh[-1]

    root_vel = vel[:, 0]
    root_local = np.stack(
        [dhead, root_vel[:, 0], root_vel[:, 2], root[:, 1]], axis=-1)

    feat = np.concatenate(
        [global_root, root_local, ric_data, rot6d, vel.reshape(T, -1),
         np.asarray(contacts, dtype=np.float64)], axis=-1)
    if feat.shape[1] != DIM:
        raise AssertionError(f"features {feat.shape[1]} != {DIM}")
    return feat


def decode_pose(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vector de rasgos -> (posiciones, rotaciones) en el frame canónico."""
    motion = np.asarray(motion, dtype=np.float64)
    T = motion.shape[0]
    if motion.shape[1] != DIM:
        raise ValueError(f"se esperaba (T,{DIM}), hay {motion.shape}")
    gr = motion[:, slice(*LAYOUT["global_root"])]
    ric = motion[:, slice(*LAYOUT["ric_data"])].reshape(T, JOINTS - 1, 3)
    r6 = motion[:, slice(*LAYOUT["global_rot_data"])].reshape(T, JOINTS, 6)
    root = gr[:, 0:3]
    pos = np.zeros((T, JOINTS, 3), dtype=np.float64)
    pos[:, 0] = root
    pos[:, 1:, 0] = ric[..., 0] + root[:, 0:1]
    pos[:, 1:, 1] = ric[..., 1]
    pos[:, 1:, 2] = ric[..., 2] + root[:, 2:3]
    rot = cont6d_to_matrix(r6)
    return pos, rot


def roundtrip_error(motion: np.ndarray) -> tuple[float, float]:
    """Ida y vuelta del vector: (error posición m, error rotación deg)."""
    pos, rot = decode_pose(motion)
    gr = motion[:, slice(*LAYOUT["global_root"])]
    heading = np.arctan2(gr[:, 4], gr[:, 3])
    contacts = motion[:, slice(*LAYOUT["foot_contacts"])]
    feat = encode_pose(pos, rot, heading, contacts)
    end = LAYOUT["foot_contacts"][0]
    pos_err = float(np.abs(feat[:, :end] - motion[:, :end]).max())
    r6 = motion[:, slice(*LAYOUT["global_rot_data"])].reshape(
        motion.shape[0], JOINTS, 6)
    rel = np.swapaxes(rot, -1, -2) @ cont6d_to_matrix(r6)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    rot_err = float(np.degrees(np.arccos(
        np.clip((tr - 1.0) / 2.0, -1.0, 1.0))).max())
    return pos_err, rot_err


def ground_pose(pos: np.ndarray, contacts_src: np.ndarray,
                smooth: float = SMOOTH_SIGMA) -> np.ndarray:
    """Offset vertical por frame: pies en el suelo, sin penetración."""
    T = pos.shape[0]
    feet = pos[:, FEET, 1]
    low = feet.min(axis=1)
    idx = np.flatnonzero(np.asarray(contacts_src).any(axis=1))
    off = np.zeros(T, dtype=np.float64)

    if idx.size == 0:
        off[:] = -np.median(low)
    else:
        off[idx] = -low[idx]
        off[: idx[0]] = off[idx[0]]
        off[idx[-1] + 1:] = off[idx[-1]]
        for a, b in zip(idx[:-1], idx[1:]):
            if b > a + 1:
                off[a + 1:b] = np.linspace(off[a], off[b], b - a + 1)[1:-1]

    if smooth > 0:
        r = int(np.ceil(3.0 * smooth))
        x = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / smooth) ** 2)
        k /= k.sum()
        off = np.convolve(np.pad(off, (r, r), mode="edge"), k, mode="valid")

    off = np.maximum(off, -low)
    return off


def detect_contacts(pos: np.ndarray, fps: float = FPS,
                    h_on: float = H_ON, h_off: float = H_OFF,
                    v_on: float = V_ON, v_off: float = V_OFF) -> np.ndarray:
    """Contactos por altura y velocidad, con histéresis por pie."""
    feet = pos[:, FEET]
    T = feet.shape[0]
    h = feet[..., 1]
    v = np.zeros_like(h)
    if T > 1:
        v[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1) * fps
        v[-1] = v[-2]
    out = np.zeros((T, len(FEET)), dtype=bool)
    for k in range(len(FEET)):
        state = False
        for t in range(T):
            if state:
                if h[t, k] > h_off or v[t, k] > v_off:
                    state = False
            else:
                if h[t, k] < h_on and v[t, k] < v_on:
                    state = True
            out[t, k] = state
    return out


# --------------------------------------------------------------------------- #
# Entrada de datos
# --------------------------------------------------------------------------- #
def load_source_asset():
    """Carga el esqueleto de origen embebido."""
    raw = base64.b64decode(_SOURCE_ASSET_B64)
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        ref = np.asarray(z["ref_local"], np.float64)
        parents = np.asarray(z["parents"], np.int64)
        names = [str(n) for n in z["joint_names"]]
        t_world = np.asarray(z["t_pose_world"], np.float64)
    n = int(ref.shape[0]) if ref.ndim else 0
    if ref.shape != (n, 7):
        raise ValueError(f"ref_local {ref.shape} != ({n},7)")
    if parents.shape != (n,):
        raise ValueError(f"parents {parents.shape} != ({n},)")
    if t_world.shape != (n, 3):
        raise ValueError(f"t_pose_world {t_world.shape} != ({n},3)")
    if len(names) != n:
        raise ValueError(f"{len(names)} nombres != {n}")
    return ref, parents, names, t_world


def load_native(path: Path) -> dict:
    """Lee un NPZ con rotaciones, traslación y confianza."""
    z = np.load(path, allow_pickle=False)
    need = ("global_orient", "body_pose", "transl", "kp2d_native77",
            "bbx_xys", "fps", "frame_count")
    missing = [k for k in need if k not in z.files]
    if missing:
        raise ValueError(f"{path}: faltan claves {missing}")
    T = int(np.asarray(z["frame_count"]).reshape(-1)[0])
    go = np.asarray(z["global_orient"], np.float64)
    bp = np.asarray(z["body_pose"], np.float64)
    if bp.ndim != 2:
        raise ValueError(f"{path}: body_pose {bp.shape} no es una matriz")
    S = bp.shape[1] // 3 + 1
    if go.shape != (T, 3):
        raise ValueError(f"{path}: global_orient {go.shape} != ({T},3)")
    if bp.shape != (T, (S - 1) * 3):
        raise ValueError(f"{path}: body_pose {bp.shape} no es (T,3*(S-1))")
    poses = np.concatenate([go[:, None], bp.reshape(T, S - 1, 3)], axis=1)
    return {
        "path": path,
        "T": T,
        "fps": float(np.asarray(z["fps"]).reshape(-1)[0]),
        "poses": poses,
        "transl": np.asarray(z["transl"], np.float64),
        "conf": np.asarray(z["kp2d_native77"], np.float64)[:, :, 2],
        "bbox": np.asarray(z["bbx_xys"], np.float64),
    }


def pt_to_npz(pt_path: str | Path, npz_path: str | Path) -> Path:
    """Extrae del .pt las claves que consume el lector."""
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    def to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) \
            else np.asarray(x)

    g = obj["pred_soma_params_global"]
    cam = obj["camera"]
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        global_orient=to_np(g["global_orient"][0]),
        body_pose=to_np(g["body_pose"][0]),
        transl=to_np(g["transl"][0]),
        identity_coeffs=to_np(g["identity_coeffs"][0]),
        scale_params=to_np(g["scale_params"][0]),
        R_w2c=to_np(cam["R_w2c"]),
        K_fullimg=to_np(obj["conditions"]["K_fullimg"][0]),
        kp2d_native77=to_np(obj["conditions"]["kp2d_native77"][0]),
        bbx_xys=to_np(obj["conditions"]["bbx_xys"][0]),
        frame_times=to_np(obj["frame_times"][0]),
        fps=np.array([obj["sampling_fps"]], dtype=np.float32),
        frame_count=np.array([int(obj["length"][0])], dtype=np.int64),
    )
    return npz_path


# --------------------------------------------------------------------------- #
# Cinemática del esqueleto de origen
# --------------------------------------------------------------------------- #
def qmat(q: np.ndarray) -> np.ndarray:
    """Cuaternión xyzw -> matriz de rotación (...,3,3)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                  2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                  2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                  1 - 2 * (x * x + y * y)], -1)], -2)


def aa2q(v: np.ndarray) -> np.ndarray:
    """Axis-angle (...,3) -> cuaternión xyzw (...,4)."""
    a = np.linalg.norm(v, axis=-1, keepdims=True)
    h = 0.5 * a
    s = np.sin(h) / np.where(a < 1e-9, 1.0, a)
    return np.concatenate([v * s, np.cos(h)], -1)


def smooth_quat(q: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Suavizado Savitzky-Golay con continuidad de signo."""
    q = q.copy()
    for t in range(1, len(q)):
        dot = np.sum(q[t] * q[t - 1], -1)
        q[t] *= np.where(dot < 0, -1.0, 1.0)[:, None]
    qs = savgol_filter(q, win, poly, axis=0, mode="interp")
    return qs / np.maximum(np.linalg.norm(qs, axis=-1, keepdims=True), 1e-12)


def frame_from(base: np.ndarray, mid: np.ndarray,
               left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Frame ortonormal (columnas: lateral, arriba, frontal)."""
    up = np.asarray(mid, float) - np.asarray(base, float)
    side = np.asarray(left, float) - np.asarray(right, float)
    up = up / np.linalg.norm(up)
    side = side / np.linalg.norm(side)
    fwd = np.cross(side, up)
    fwd = fwd / np.linalg.norm(fwd)
    side = np.cross(up, fwd)
    return np.stack([side, up, fwd], -1)


def jitter_deg(rot: np.ndarray) -> np.ndarray:
    """Ángulo (deg) entre rotaciones globales consecutivas (T-1, J)."""
    rel = np.swapaxes(rot[:-1], -1, -2) @ rot[1:]
    tr = np.trace(rel, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def skate_mps(pos: np.ndarray, contacts: np.ndarray,
              feet: tuple[int, ...], fps: float) -> np.ndarray:
    """Velocidad horizontal (m/s) de cada pie en sus frames de contacto."""
    T = pos.shape[0]
    v = np.zeros((T, len(feet)))
    for k, j in enumerate(feet):
        d = np.linalg.norm(pos[1:, j, [0, 2]] - pos[:-1, j, [0, 2]], axis=-1)
        v[:-1, k] = d * fps
        v[-1, k] = v[-2, k]
    return v[contacts]


# --------------------------------------------------------------------------- #
# Suavizado, contactos y transferencia
# --------------------------------------------------------------------------- #
SMOOTH_WIN = 5
SMOOTH_POLY = 2
CONF_MIN = 0.60
FLOOR_WIN = 61
FLOOR_PCT = 10.0

H_ON_SRC, H_OFF_SRC = 0.08, 0.12
V_ON_SRC, V_OFF_SRC = 1.00, 2.00

BONES = [
    (1, 4, 67, 68), (4, 5, 68, 69), (5, 7, 69, 70),
    (8, 11, 72, 73), (11, 12, 73, 74), (12, 14, 74, 75),
    (18, 21, 12, 13), (21, 22, 13, 14),
    (26, 29, 40, 41), (29, 30, 41, 42),
]
INTERMEDIATE = (2, 3, 6, 9, 10, 13, 19, 20, 22, 23, 24,
                27, 28, 30, 31, 32)
SOURCE_FEET = (69, 70, 74, 75)


def shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotación (...,3,3) que lleva `a` a `b` por el arco mínimo."""
    def normalize(v, eps=1e-8):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(n, eps)

    def skew(v):
        z = np.zeros_like(v[..., 0])
        return np.stack([
            np.stack([z, -v[..., 2], v[..., 1]], axis=-1),
            np.stack([v[..., 2], z, -v[..., 0]], axis=-1),
            np.stack([-v[..., 1], v[..., 0], z], axis=-1),
        ], axis=-2)

    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = np.sum(a * b, axis=-1)
    s = np.linalg.norm(v, axis=-1)

    eye = np.broadcast_to(np.eye(3, dtype=a.dtype),
                          a.shape[:-1] + (3, 3)).copy()
    out = eye.copy()

    general = s > 1e-7
    if np.any(general):
        axis = v[general] / s[general][..., None]
        K = skew(axis)
        angle = np.arctan2(s[general], c[general])
        sin = np.sin(angle)[..., None, None]
        cos = np.cos(angle)[..., None, None]
        out[general] = eye[general] + sin * K + (1.0 - cos) * (K @ K)

    anti = (~general) & (c <= 0)
    if np.any(anti):
        av = a[anti]
        pick = np.argmin(np.abs(av), axis=-1)
        e = np.zeros_like(av)
        e[np.arange(len(av)), pick] = 1.0
        axis = normalize(np.cross(av, e))
        K = skew(axis)
        out[anti] = eye[anti] + 2.0 * (K @ K)
    return out


def source_fk(poses: np.ndarray, root: np.ndarray, ref: np.ndarray,
              parents: np.ndarray, t_world: np.ndarray,
              smooth: bool = True):
    """Cinemática directa del esqueleto de origen."""
    T = poses.shape[0]
    S = len(parents)
    if poses.ndim != 3 or poses.shape[2] != 3:
        raise ValueError(f"poses {poses.shape} no es (T,{S},3)")
    if poses.shape[1] != S:
        raise ValueError(f"poses {poses.shape} != ({T},{S},3)")
    Rl = qmat(ref[:, 3:7])
    Rtp = np.zeros((S, 3, 3))
    Rtp[0] = Rl[0]
    Ptp = np.zeros((S, 3))
    Ptp[0] = ref[0, :3]
    for j in range(1, S):
        p = int(parents[j])
        Rtp[j] = Rtp[p] @ Rl[j]
        Ptp[j] = Ptp[p] + Rtp[p] @ ref[j, :3]
    if not np.allclose(Ptp, t_world, atol=1e-9):
        raise AssertionError("la pose de reposo no coincide con la FK")

    q = aa2q(poses)
    if smooth:
        q = smooth_quat(q, SMOOTH_WIN, SMOOTH_POLY)
    Rrel = qmat(q)
    A = np.zeros((T, S, 3, 3))
    A[:, 0] = Rrel[:, 0]
    for j in range(1, S):
        A[:, j] = A[:, int(parents[j])] @ Rrel[:, j]
    P = np.zeros((T, S, 3))
    P[:, 0] = root
    for j in range(1, S):
        p = int(parents[j])
        P[:, j] = P[:, p] + np.einsum("tij,j->ti", A[:, p],
                                      Ptp[j] - Ptp[p])
    return P, A, Ptp, Rtp


def detect_source_contacts(P: np.ndarray, fps: float):
    """Contactos relativos a la cadera, robustos a deriva vertical."""
    idx = np.array(SOURCE_FEET)
    rel = P[:, idx, 1] - P[:, 0:1, 1]
    vel = np.gradient(P, axis=0) * fps
    vrel = np.linalg.norm(vel[:, idx] - vel[:, 0:1], axis=-1)
    low = rel.min(axis=1)
    floor = percentile_filter(low, size=FLOOR_WIN, percentile=FLOOR_PCT,
                              mode="nearest")
    h = rel - floor[:, None]
    out = np.zeros_like(h, dtype=bool)
    for k in range(len(SOURCE_FEET)):
        state = False
        for t in range(len(h)):
            if state:
                if h[t, k] > H_OFF_SRC or vrel[t, k] > V_OFF_SRC:
                    state = False
            elif h[t, k] < H_ON_SRC and vrel[t, k] < V_ON_SRC:
                state = True
            out[t, k] = state
    return out, {"height": h, "vrel": vrel, "floor": floor}


def map_pose(A: np.ndarray, Ptp: np.ndarray, rig: np.ndarray,
             rigp: np.ndarray, root: np.ndarray):
    """Rotaciones globales con dirección de hueso exacta y twist suave."""
    T = A.shape[0]
    N = len(rig)
    F_d_root = frame_from(rig[0], rig[17], rig[1], rig[8])
    F_s_root = frame_from(Ptp[0], Ptp[3], Ptp[67], Ptp[72])
    F_d_chest = frame_from(rig[17], 0.5 * (rig[18] + rig[26]),
                           rig[18], rig[26])
    F_s_chest = frame_from(Ptp[3], Ptp[3] + 0.25 * (Ptp[3] - Ptp[0]),
                           Ptp[11], Ptp[39])

    Rg = np.broadcast_to(np.eye(3), (T, N, 3, 3)).copy()
    Rg[:, 0] = A[:, 0] @ (F_s_root @ F_d_root.T)
    Rg[:, 15] = Rg[:, 16] = Rg[:, 17] = (
        A[:, 3] @ (F_s_chest @ F_d_chest.T))

    for dj, dtip, sj, stip in BONES:
        ud = rig[dtip] - rig[dj]
        us = Ptp[stip] - Ptp[sj]
        M = shortest_arc(ud / np.linalg.norm(ud),
                         us / np.linalg.norm(us))
        Rg[:, dj] = A[:, sj] @ M

    for j in INTERMEDIATE:
        Rg[:, j] = Rg[:, int(rigp[j])]
    Rg[:, 7] = Rg[:, 5]
    Rg[:, 14] = Rg[:, 12]
    Rg[:, 25] = Rg[:, 21]
    Rg[:, 33] = Rg[:, 29]
    for j in DEAD_LEAVES:
        Rg[:, j] = np.eye(3)

    Pg = fk(rig, rigp, root, Rg)
    return Pg, Rg


def valid_prefix(conf: np.ndarray, bbox: np.ndarray) -> tuple[int, str]:
    """Último frame válido (recorta la cola vacía)."""
    c = conf.mean(axis=1)
    valid = c >= CONF_MIN
    for t in range(1, len(valid)):
        if np.allclose(bbox[t], bbox[t - 1], atol=1e-6) and c[t] < 0.8:
            valid[t] = False
    last = len(valid) - 1
    while last >= 0 and not valid[last]:
        last -= 1
    if last + 1 < 30:
        raise ValueError(f"solo {last + 1} frames válidos (<30)")
    holes = []
    t = 0
    while t < last + 1:
        if not valid[t]:
            s = t
            while t <= last and not valid[t]:
                t += 1
            if t - s >= 5:
                holes.append([int(s), int(t - 1)])
        else:
            t += 1
    reason = (f"conf media >= {CONF_MIN:.2f} y bbox no congelada; "
              f"cola descartada {last + 1}:{len(valid)}")
    if holes:
        reason += f"; huecos internos >=5 frames: {holes}"
    return last + 1, reason


# --------------------------------------------------------------------------- #
# BVH
# --------------------------------------------------------------------------- #
ROOT_NAME = "Root"
FRAME_TIME = 1.0 / 30.0
CH_ROOT = "6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
CH_JOINT = "3 Zrotation Xrotation Yrotation"


def _rot_zxy_deg(rot: np.ndarray) -> np.ndarray:
    """Matrices (...,3,3) -> Euler ZXY en grados."""
    return Rot.from_matrix(np.asarray(rot, dtype=np.float64)).as_euler(
        "ZXY", degrees=True)


def _unwrap_deg(eul: np.ndarray) -> np.ndarray:
    """Quita los saltos de ±180° de los Euler a lo largo del tiempo."""
    out = np.array(eul, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        for k in range(out.shape[2]):
            out[:, j, k] = np.unwrap(out[:, j, k], period=360, discont=180)
    return out


def foot_link(rest: np.ndarray, names: list[str]) -> float:
    """Altura de la punta del pie bajo la pelvis en reposo."""
    toes = [j for j, n in enumerate(names) if n.endswith("toe_base")]
    if not toes:
        toes = [j for j, n in enumerate(names) if "toe" in n]
    if not toes:
        return 0.0
    return float(-np.mean(rest[toes, 1]))


def write_bvh(path: Path, pos: np.ndarray, rot: np.ndarray,
              rest: np.ndarray, parents: np.ndarray, names: list[str],
              fps: float = FPS) -> dict:
    """Escribe un BVH con ancla Root, orden ZXY y centímetros."""
    pos = np.asarray(pos, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    rest = np.asarray(rest, dtype=np.float64)
    par = np.asarray(parents)
    T, J = pos.shape[0], len(names)

    children: dict[int, list[int]] = {j: [] for j in range(J)}
    for j in range(J):
        p = int(par[j])
        if p >= 0:
            children[p].append(j)

    order: list[int] = []

    def dfs(j: int) -> None:
        order.append(j)
        for c in children[j]:
            dfs(c)

    dfs(0)

    link = foot_link(rest, names)
    rloc = np.zeros_like(rot)
    for j in range(J):
        p = int(par[j])
        if p < 0:
            rloc[:, j] = rot[:, j]
        else:
            rloc[:, j] = np.swapaxes(rot[:, p], -1, -2) @ rot[:, j]
    eul = _unwrap_deg(_rot_zxy_deg(rloc))

    anchor_cm = np.stack([pos[:, 0, 0],
                          pos[:, 0, 1] - link,
                          pos[:, 0, 2]], axis=1) * 100.0

    def node(j: int):
        p = int(par[j])
        off = (rest[j] - rest[p]) * 100.0 if p >= 0 else np.zeros(3)
        kids = [node(c) for c in children[j]]
        return [names[j], off, CH_JOINT, kids]

    pelvis = node(0)
    pelvis[1] = np.array([0.0, link * 100.0, 0.0])
    root = [ROOT_NAME, np.zeros(3), CH_ROOT, [pelvis]]

    lines: list[str] = ["HIERARCHY"]

    def emit(nd, depth: int) -> None:
        pad = "  " * depth
        tag = "ROOT" if depth == 0 else "JOINT"
        nm, off, ch, kids = nd
        lines.append(f"{pad}{tag} {nm}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        lines.append(f"{pad}  CHANNELS {ch}")
        for k in kids:
            emit(k, depth + 1)
        if not kids:
            lines.append(f"{pad}  End Site")
            lines.append(f"{pad}  {{")
            lines.append(f"{pad}    OFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")

    emit(root, 0)
    lines.extend(["MOTION", f"Frames: {T}",
                  f"Frame Time: {FRAME_TIME:.8f}"])
    for t in range(T):
        vals = [*anchor_cm[t], 0.0, 0.0, 0.0]
        for j in order:
            vals.extend(eul[t, j])
        lines.append(" ".join(f"{v:.6f}" for v in vals))

    path.write_text("\n".join(lines) + "\n")
    return {
        "frames": int(T),
        "joints": int(J),
        "link_m": link,
        "anchor_y_min_m": float(anchor_cm[:, 1].min() / 100.0),
        "anchor_y_max_m": float(anchor_cm[:, 1].max() / 100.0),
        "frame_time": FRAME_TIME,
    }


def _rot_axis(axis: str, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def read_bvh(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Lee un BVH y devuelve nombres, posiciones (cm) y rotaciones."""
    txt = path.read_text(errors="replace")
    htxt, mtxt = txt.split("MOTION", 1)
    lines = [ln.strip() for ln in htxt.splitlines() if ln.strip()]

    nodes: list[dict] = []

    def parse(i: int, parent: int) -> int:
        tag, name = lines[i].split()
        assert tag in ("ROOT", "JOINT"), tag
        i += 1
        assert lines[i] == "{", lines[i]
        i += 1
        node = {"name": name, "parent": parent, "off": np.zeros(3),
                "ch": [], "idx": len(nodes)}
        nodes.append(node)
        while lines[i] != "}":
            s = lines[i]
            if s.startswith("OFFSET"):
                node["off"] = np.array([float(v) for v in s.split()[1:4]])
                i += 1
            elif s.startswith("CHANNELS"):
                n = int(s.split()[1])
                node["ch"] = s.split()[2:2 + n]
                i += 1
            elif s == "End Site":
                i += 1
                i += 1
                i += 1
                i += 1
            elif s.startswith("ROOT") or s.startswith("JOINT"):
                i = parse(i, node["idx"])
            else:
                i += 1
        return i + 1

    i = 0
    while i < len(lines):
        if lines[i].startswith("ROOT") or lines[i].startswith("JOINT"):
            i = parse(i, -1)
        else:
            i += 1

    body = mtxt.split("Frame Time:", 1)[1]
    body = re.sub(r"^\s*[-+0-9.eE]+", "", body, count=1)
    vals = [float(v) for v in body.split()]
    nch = sum(len(n["ch"]) for n in nodes)
    T = len(vals) // nch
    arr = np.array(vals[:T * nch]).reshape(T, nch)

    pos = np.zeros((T, len(nodes), 3))
    rot = np.zeros((T, len(nodes), 3, 3))
    for t in range(T):
        col = 0
        for node in nodes:
            ch = node["ch"]
            trans = np.zeros(3)
            if len(ch) == 6:
                trans = arr[t, col:col + 3]
                col += 3
            angs = arr[t, col:col + 3]
            col += 3
            M = np.eye(3)
            for c, a in zip(ch, angs):
                M = M @ _rot_axis(c[0], np.radians(a))
            p = node["parent"]
            if p < 0:
                rot[t, node["idx"]] = M
                pos[t, node["idx"]] = trans
            else:
                rot[t, node["idx"]] = rot[t, p] @ M
                pos[t, node["idx"]] = (pos[t, p] +
                                       rot[t, p] @ node["off"])
    return [n["name"] for n in nodes], pos, rot


def verify_bvh(path: Path, pos: np.ndarray, rot: np.ndarray,
               names: list[str], feet: tuple[int, ...]) -> dict:
    """Relee el BVH y lo compara con la pose de referencia."""
    bnames, bpos, brot = read_bvh(path)
    if bnames[1:] != list(names):
        raise AssertionError("el BVH no tiene la jerarquía esperada")
    bp = bpos[:, 1:] / 100.0
    br = brot[:, 1:]
    perr = float(np.abs(bp - pos).max())
    rel = np.swapaxes(rot, -1, -2) @ br
    tr = np.trace(rel, axis1=-2, axis2=-1)
    rerr = float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0,
                                               -1.0, 1.0))).max())
    return {
        "frames": int(bp.shape[0]),
        "joints": int(bp.shape[1]),
        "pos_max_err_m": perr,
        "rot_max_err_deg": rerr,
        "min_foot_y": float(bp[:, list(feet), 1].min()),
        "root_y_min": float(bp[:, 0, 1].min()),
        "root_y_max": float(bp[:, 0, 1].max()),
        "root_xz_net_m": float(np.linalg.norm(bp[-1, 0, [0, 2]] -
                                             bp[0, 0, [0, 2]])),
    }


# --------------------------------------------------------------------------- #
# Núcleo de conversión
# --------------------------------------------------------------------------- #
def convert_arrays(poses: np.ndarray, root: np.ndarray, fps: float,
                   ref: np.ndarray, parents: np.ndarray,
                   t_world: np.ndarray, rig: np.ndarray, rigp: np.ndarray,
                   *, name: str, src: Path, out_dir: Path,
                   smooth: bool = True, verbose: bool = True) -> dict:
    """Ejecuta el proceso completo y escribe NPZ + BVH."""
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    out_dir.mkdir(parents=True, exist_ok=True)

    P, A, tw, _ = source_fk(poses, root, ref, parents, t_world,
                            smooth=smooth)
    contacts_src, _ = detect_source_contacts(P, fps)

    Pg, Rg = map_pose(A, tw, rig, rigp, root)
    off = ground_pose(Pg, contacts_src, smooth=SMOOTH_SIGMA)
    Pg = Pg.copy()
    Pg[:, :, 1] += off[:, None]
    contacts = detect_contacts(Pg, fps)

    Pc, Rc, hc = canonicalize_pose(Pg, Rg)
    motion = encode_pose(Pc, Rc, hc, contacts, fps)
    if not np.isfinite(motion).all():
        raise AssertionError("datos no finitos")

    P2, R2 = decode_pose(motion)
    pos_err = float(np.abs(P2 - Pc).max())
    rel = np.swapaxes(Rc, -1, -2) @ R2
    tr = np.trace(rel, axis1=-2, axis2=-1)
    rot_err = float(np.degrees(
        np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))).max())
    rt_err = roundtrip_error(motion)
    feet_y = P2[:, FEET, 1]
    gmin = float(feet_y.min())
    cv = bone_cv(Pg, rigp)
    if pos_err > 1e-5 or rot_err > 1e-3:
        raise AssertionError(f"ida y vuelta ({pos_err:.2e} m, "
                             f"{rot_err:.2e} deg)")
    if max(rt_err) > 1e-5:
        raise AssertionError(f"roundtrip {rt_err}")
    if gmin < -1e-5:
        raise AssertionError(f"pie bajo el suelo {gmin}")

    dst = out_dir / f"{name}.npz"
    motion_f32 = motion.astype(np.float32)
    np.savez_compressed(
        dst,
        motion=motion_f32,
        contacts_src=contacts_src.astype(np.bool_),
        fps=np.float32(fps),
        name=name,
        source=src.name,
    )

    bvh_path = out_dir / f"{name}.bvh"
    w = write_bvh(bvh_path, Pc, Rc, rig, rigp, list(NAMES), fps=fps)
    v = verify_bvh(bvh_path, Pc, Rc, list(NAMES), tuple(FEET))

    checks = {
        "finite": bool(np.isfinite(motion).all()),
        "roundtrip_pos_ok": bool(pos_err <= 1e-5),
        "roundtrip_rot_ok": bool(rot_err <= 1e-3),
        "ground_ok": bool(gmin >= -1e-5),
        "bones_ok": bool(cv < 1e-10),
        "bvh_ok": bool(v["pos_max_err_m"] <= 1e-4 and
                       v["rot_max_err_deg"] <= 1e-2 and
                       v["min_foot_y"] >= -1e-5),
    }
    checks["all_passed"] = all(checks.values())

    say(f"[datos] {dst.name}  {motion.shape[0]} frames  "
        f"{motion.shape[1]} dims")
    say(f"[bvh]   {bvh_path.name}  error {v['pos_max_err_m']:.1e} m / "
        f"{v['rot_max_err_deg']:.1e} deg  "
        f"pie mínimo {v['min_foot_y']:.6f} m")
    say(f"[fin]   {time.time() - t0:.1f} s  "
        f"{'OK' if checks['all_passed'] else 'REVISAR'}")

    return {
        "ok": bool(checks["all_passed"]),
        "checks": checks,
        "frames": int(motion.shape[0]),
        "name": name,
        "npz": dst,
        "bvh_file": bvh_path,
        "bvh": w,
    }


# --------------------------------------------------------------------------- #
# Entrada y salida
# --------------------------------------------------------------------------- #
def npz_keys(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as z:
        return list(z.files)


def safe_name(text: str) -> str:
    """Nombre estable: solo [A-Za-z0-9_.-]."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "video"


def output_stem(path: Path) -> str:
    return safe_name(path.stem)


SOURCES = (
    {
        "id": "native",
        "keys": ("global_orient", "body_pose", "transl",
                 "kp2d_native77", "bbx_xys", "fps", "frame_count"),
        "load": load_native,
    },
)


def find_source(keys: list[str]) -> dict | None:
    for spec in SOURCES:
        if all(k in keys for k in spec["keys"]):
            return spec
    return None


_ASSETS: dict | None = None


def assets() -> dict:
    global _ASSETS
    if _ASSETS is None:
        ref, parents, names, t_world = load_source_asset()
        _ASSETS = {
            "ref": ref,
            "parents": np.asarray(parents, np.int64),
            "names": [str(n) for n in names],
            "t_world": t_world,
            "rig": load_rest(),
            "rigp": np.asarray(PARENTS),
        }
    return _ASSETS


def output_name(src: Path, existing: set[str] | None = None) -> str:
    """Nombre base; si colisiona con otro vídeo del lote, sufijo estable."""
    base = output_stem(src)
    if not existing or base not in existing:
        return base
    tag = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:6]
    return f"{base}_{tag}"


def convert_file(src: str | Path, out_dir: str | Path, *,
                 name: str | None = None, smooth: bool = True,
                 verbose: bool = True) -> dict:
    """Un NPZ -> `<name>.npz` + `<name>.bvh` en `out_dir`."""
    src = Path(src).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    spec = find_source(npz_keys(src))
    if spec is None:
        raise ValueError(f"{src}: formato no reconocido")

    d = spec["load"](src)
    end, _ = valid_prefix(d["conf"], d["bbox"])
    a = assets()
    out_name = name or output_stem(src)

    return convert_arrays(
        poses=d["poses"][:end],
        root=d["transl"][:end],
        fps=d["fps"],
        ref=a["ref"], parents=a["parents"], t_world=a["t_world"],
        rig=a["rig"], rigp=a["rigp"],
        name=out_name, src=src, out_dir=out_dir,
        smooth=smooth, verbose=verbose,
    )


def convert_pt(pt_path: str | Path, out_dir: str | Path, *,
               name: str | None = None, smooth: bool = True,
               verbose: bool = True) -> dict:
    """Un .pt -> NPZ + BVH; el .pt no se conserva."""
    pt_path = Path(pt_path).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="gem_raw_") as tmp:
        raw_name = f"{name or pt_path.stem}.npz"
        raw_npz = pt_to_npz(pt_path, Path(tmp) / raw_name)
        return convert_file(raw_npz, out_dir, name=name, smooth=smooth,
                            verbose=verbose)


def convert_many(paths, out_dir: str | Path, *,
                 smooth: bool = True, verbose: bool = True) -> list[dict]:
    """Varios NPZ -> una sola carpeta, nombres únicos dentro del lote."""
    paths = [Path(p).expanduser().resolve() for p in paths]
    out_dir = Path(out_dir).expanduser().resolve()
    names: set[str] = set()
    queue = []
    for src in paths:
        spec = find_source(npz_keys(src))
        if spec is None:
            raise ValueError(f"{src}: formato no reconocido")
        name = output_name(src, names)
        names.add(name)
        queue.append((src, name))
    return [convert_file(src, out_dir, name=name, smooth=smooth,
                         verbose=verbose)
            for src, name in queue]


# --------------------------------------------------------------------------- #
# Preparación del vídeo
# --------------------------------------------------------------------------- #
def parse_rate(text) -> float | None:
    """'30000/1001' -> 29.97; '0/0' -> None."""
    if not text:
        return None
    if "/" in str(text):
        num, den = str(text).split("/", 1)
        try:
            num, den = float(num), float(den)
        except ValueError:
            return None
        return num / den if den else None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def scale_box(width: int, height: int, *,
              max_w: int = MAX_W, max_h: int = MAX_H) -> tuple[int, int]:
    """Caja de destino que quepa en max_w x max_h sin deformar."""
    factor = min(max_w / width, max_h / height, 1.0)
    w = max(2, int(round(width * factor)) // 2 * 2)
    h = max(2, int(round(height * factor)) // 2 * 2)
    return w, h


def fit_plan(width: int, height: int, fps: float | None, *,
             max_w: int = MAX_W, max_h: int = MAX_H,
             max_fps: float = MAX_FPS) -> dict:
    """Decide escala y fps; no upscalea ni duplica frames."""
    scale = width > max_w or height > max_h
    cap_fps = fps is not None and fps > max_fps + 1e-6
    out_w, out_h = scale_box(width, height, max_w=max_w, max_h=max_h)
    return {
        "scale": scale,
        "fps": cap_fps,
        "changed": scale or cap_fps,
        "width": int(width),
        "height": int(height),
        "out_width": out_w,
        "out_height": out_h,
        "fps_in": fps,
        "fps_out": max_fps if cap_fps else fps,
    }


def probe(path: str | Path) -> dict:
    """Resolución y fps reales (aplica rotación de metadatos)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=width,height,avg_frame_rate,r_frame_rate:stream_tags=rotate"
           ":stream_side_data=rotation",
           "-of", "json", str(path)]
    data = json.loads(subprocess.check_output(cmd, text=True))
    stream = (data.get("streams") or [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = (parse_rate(stream.get("avg_frame_rate"))
           or parse_rate(stream.get("r_frame_rate")))

    rotation = 0.0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = float(side["rotation"])
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        rotation = float(tags["rotate"])
    if abs(int(round(rotation))) % 180 == 90:
        width, height = height, width

    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: resolución no detectada")
    return {"width": width, "height": height, "fps": fps,
            "rotation": rotation}


def rewrite_video(src: str | Path, dst: str | Path, p: dict) -> Path:
    """Aplica el plan con una sola pasada de ffmpeg."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    width, height = scale_box(p["width"], p["height"])
    filters = [f"scale={width}:{height}:flags=lanczos",
               "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    if p["fps"]:
        filters.insert(1, f"fps={MAX_FPS:g}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-map", "0:v:0", "-vf", ",".join(filters),
           "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
           "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
           str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def prepare_video(src: str | Path, tmp_dir: str | Path) -> dict:
    """Devuelve el vídeo a usar (original o ajustado) y su plan."""
    src = Path(src)
    info = probe(src)
    p = fit_plan(info["width"], info["height"], info["fps"])
    if not p["changed"]:
        return {"path": src, "info": info, "plan": p}
    _, out_h = scale_box(p["width"], p["height"])
    dst = Path(tmp_dir) / f"{src.stem}_{out_h}p.mp4"
    rewrite_video(src, dst, p)
    return {"path": dst, "info": info, "plan": p}


# --------------------------------------------------------------------------- #
# Instalación del motor
# --------------------------------------------------------------------------- #
def run(cmd, **kwargs) -> None:
    subprocess.run(cmd, check=True, **kwargs)


def clone_engine() -> None:
    if ENGINE_ROOT.is_dir():
        return
    print("🚀 Preparando motor...")
    run(["git", "clone", "--recursive",
         "https://github.com/NVlabs/GEM-X.git", str(ENGINE_ROOT)])


def install_python() -> None:
    print("📦 Instalando paquetes base...")
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[video]"],
        cwd=str(ENGINE_ROOT))
    print("🔧 Configurando runtime GPU...")
    run([sys.executable, "-m", "pip", "uninstall", "-y",
         "onnxruntime", "onnxruntime-gpu"], capture_output=True)
    run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
         "--no-deps", "--no-cache-dir", f"onnxruntime-gpu=={ONNX_VERSION}"])
    run([sys.executable, "-m", "pip", "install", "-q", *PIP_EXTRA])


def download_assets() -> None:
    from huggingface_hub import hf_hub_download
    from shutil import copyfileobj

    print("⬇️ Descargando pesos y assets (esto tarda un poco)...")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    (SOURCE_ASSETS / "MHR").mkdir(parents=True, exist_ok=True)
    for repo_id, revision, fname, dest in HF_FILES:
        kwargs = {"repo_id": repo_id, "filename": fname, "local_dir": dest}
        if revision:
            kwargs["revision"] = revision
        hf_hub_download(**kwargs)

    yolox_onnx = WEIGHTS / "yolox_humanart.onnx"
    if not yolox_onnx.exists():
        url = ("https://download.openmmlab.com/mmpose/v1/projects/"
               "rtmposev1/onnx_sdk/"
               "yolox_x_8xb8-300e_humanart-a39d44ed.zip")
        urllib.request.urlretrieve(url, WEIGHTS / "yolox.zip")
        with zipfile.ZipFile(WEIGHTS / "yolox.zip") as z:
            member = next(n for n in z.namelist()
                          if n.endswith("/end2end.onnx"))
            with z.open(member) as src, yolox_onnx.open("wb") as dst:
                copyfileobj(src, dst)


def configure_yaml() -> None:
    print("⚙️ Configurando parámetros...")
    yaml_path = WEIGHTS / "body_video_native77.yaml"
    txt = yaml_path.read_text().replace("device: cpu", "device: cuda")
    txt = re.sub(r"(pose:\n(?:  .*\n)*?  batch_size: )\d+", r"\g<1>32", txt)
    txt = re.sub(r"(observations:\n(?:  .*\n)*?  batch_size: )\d+",
                 r"\g<1>32", txt)
    yaml_path.write_text(txt)


def nvidia_lib_dirs() -> list[str]:
    """Descubre las librerías de nvidia instaladas."""
    import sysconfig

    roots = {sysconfig.get_paths()["purelib"],
             sysconfig.get_paths()["platlib"]}
    out = []
    for root in roots:
        base = Path(root) / "nvidia"
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            lib = sub / "lib"
            if lib.is_dir():
                out.append(str(lib))
    return out


def configure_runtime() -> None:
    import torch

    libs = [os.path.join(os.path.dirname(torch.__file__), "lib")]
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        libs + nvidia_lib_dirs() + [os.environ.get("LD_LIBRARY_PATH", "")])


# --------------------------------------------------------------------------- #
# Procesamiento
# --------------------------------------------------------------------------- #
def process_video(video: str | Path, name: str | None = None) -> dict:
    """Un vídeo -> `<name>.npz` + `<name>.bvh` en OUTPUT_DIR."""
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    stem = safe_name(video.stem)
    out_name = name or stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n🎥 Procesando vídeo: {video}")
    with tempfile.TemporaryDirectory(prefix=f"gem_{stem}_", dir=WORK) as tmp:
        prep = prepare_video(video, tmp)
        q = prep["plan"]
        if q["changed"]:
            print(f"📐 Calidad: {q['width']}x{q['height']}@{q['fps_in']} -> "
                  f"{q['out_width']}x{q['out_height']}@{q['fps_out']}")
        else:
            print(f"📐 Calidad: {q['width']}x{q['height']}@{q['fps_in']} "
                  f"(sin cambios)")

        raw_pt = Path(tmp) / f"{stem}_raw.pt"
        cmd = ["gem-infer", "motion-video", "--task", "body",
               "--input", str(prep["path"]),
               "--config", str(WEIGHTS / "body_video_native77.yaml"),
               "--device", DEVICE, "--camera-mode", CAMERA,
               "--postprocess", "--soma-assets", str(SOURCE_ASSETS),
               "--output", str(raw_pt)]
        t0 = time.time()
        run(cmd, cwd=str(ENGINE_ROOT))
        print(f"✅ Proceso: {(time.time() - t0) / 60:.2f} min.")

        result = convert_pt(raw_pt, OUTPUT_DIR, name=out_name, smooth=True,
                            verbose=True)

    print(f"🏆 LISTO: {result['npz'].name} + {result['bvh_file'].name}")
    return {
        "video": str(video),
        "name": out_name,
        "npz": str(result["npz"]),
        "bvh": str(result["bvh_file"]),
        "frames": result["frames"],
        "ok": bool(result["ok"]),
        "checks": result["checks"],
        "quality": q,
    }


def process_videos(videos) -> list[dict]:
    """Procesa todos los vídeos en una sola carpeta de salida."""
    videos = [Path(v).expanduser().resolve() for v in videos]
    names: set[str] = set()
    queue = []
    for video in videos:
        name = output_name(video, names)
        names.add(name)
        queue.append((video, name))

    results = []
    for video, name in queue:
        try:
            results.append(process_video(video, name=name))
        except Exception as exc:
            print(f"❌ {video.name}: {type(exc).__name__}: {exc}")
            results.append({"video": str(video), "name": name,
                            "ok": False, "error": str(exc)})

    ok = sum(bool(r.get("ok")) for r in results)
    print(f"\n📊 Lote: {ok}/{len(results)} vídeos -> {OUTPUT_DIR}")
    return results


# --------------------------------------------------------------------------- #
# Celda 2: varias rutas entre comillas, sin importar el formato
# --------------------------------------------------------------------------- #
def strings_in_cell(raw: str) -> list[str]:
    """Todas las cadenas literales de la celda, en orden."""
    out: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(raw).readline)
        for tok in tokens:
            if tok.type != tokenize.STRING:
                continue
            try:
                value = ast.literal_eval(tok.string)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, str):
                out.append(value)
    except (tokenize.TokenError, IndentationError):
        pass
    return out


def strings_in_result(value) -> list[str]:
    """Recorre el resultado de la celda (str, list, tuple, dict)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(strings_in_result(item))
        return out
    return []


def video_paths(raw: str = "", result=None) -> list[str]:
    """Rutas de vídeo existentes dentro de las comillas de la celda."""
    candidates = strings_in_cell(raw) + strings_in_result(result)
    seen, out = set(), []
    for text in candidates:
        path = Path(text.strip()).expanduser()
        if path.suffix.lower() not in VIDEO_SUFFIXES or not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def auto_execute(result) -> None:
    raw = getattr(getattr(result, "info", None), "raw_cell", "") or ""
    paths = video_paths(raw, getattr(result, "result", None))
    if not paths:
        return
    print(f"✨ {len(paths)} vídeo(s) detectado(s) en la celda.")
    process_videos(paths)


def register_cell_watcher() -> None:
    ip = get_ipython()
    if ip is None:
        return
    callbacks = ip.events.callbacks.get("post_run_cell", [])
    ip.events.callbacks["post_run_cell"] = [
        cb for cb in callbacks if cb.__name__ != "auto_execute"]
    ip.events.register("post_run_cell", auto_execute)


# --------------------------------------------------------------------------- #
# Arranque
# --------------------------------------------------------------------------- #
def main() -> None:
    warnings.filterwarnings("ignore")
    clone_engine()
    os.chdir(ENGINE_ROOT)
    install_python()
    download_assets()
    configure_yaml()
    configure_runtime()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register_cell_watcher()
    print(f"\n✅ ENTORNO LISTO. Salida: {OUTPUT_DIR}")
    print("   Celda 2: pon una o varias rutas de vídeo entre comillas y "
          "dale Play.")


if __name__ == "__main__":
    main()
