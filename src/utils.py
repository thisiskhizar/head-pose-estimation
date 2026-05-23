"""Shared helpers for head-pose feature extraction, training, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_INDICES: dict[str, int] = {
    "NOSE": 1,
    "FOREHEAD": 10,
    "LEFT_EYE": 33,
    "MOUTH_LEFT": 61,
    "CHIN": 199,
    "RIGHT_EYE": 263,
    "MOUTH_RIGHT": 291,
}
FEATURE_PREFIXES: tuple[str, ...] = tuple(name.lower() + "_" for name in FEATURE_INDICES)
POSE_COLUMNS: tuple[str, str, str] = ("pitch", "yaw", "roll")


def init_face_mesh(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp.solutions.face_mesh.FaceMesh:
    """Initialize MediaPipe FaceMesh with the notebook confidence defaults."""
    return mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def generate_feature_column_names(
    include_pose: bool = True,
    selected_points: Mapping[str, int] = FEATURE_INDICES,
) -> list[str]:
    """Return landmark feature columns, optionally followed by pose-angle columns."""
    columns = [
        f"{name.lower()}_{dim}"
        for name in selected_points
        for dim in ("x", "y")
    ]
    if include_pose:
        columns.extend(POSE_COLUMNS)
    return columns


def load_rgb_image(image_path: str | Path) -> tuple[np.ndarray, int, int]:
    """Load an image from disk as RGB and return it with width and height."""
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image could not be loaded: {path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w = image.shape[:2]
    return image_rgb, image_w, image_h


def load_pose_angles(mat_path: str | Path) -> np.ndarray:
    """Load pitch, yaw, and roll from an AFLW2000 `.mat` annotation file."""
    mat = scipy.io.loadmat(str(mat_path))
    return np.asarray(mat["Pose_Para"][0][:3], dtype=float)


def load_image_and_pose(
    img_idx: int,
    img_dir: str | Path = "./data/raw/AFLW2000",
    mat_ext: str = ".mat",
    img_ext: str = ".jpg",
) -> tuple[np.ndarray, float, float, float]:
    """Load one AFLW2000 image and its pitch, yaw, and roll by dataset index."""
    img_dir = Path(img_dir)
    img_paths = sorted(img_dir.glob(f"*{img_ext}"))
    mat_paths = sorted(img_dir.glob(f"*{mat_ext}"))
    try:
        image_rgb, _, _ = load_rgb_image(img_paths[img_idx])
        pitch, yaw, roll = load_pose_angles(mat_paths[img_idx])
    except IndexError as exc:
        raise FileNotFoundError(
            "Image or corresponding .mat file not found for the given index."
        ) from exc
    return image_rgb, float(pitch), float(yaw), float(roll)


def pair_image_and_mat_paths(
    image_paths: Iterable[str | Path],
    mat_paths: Iterable[str | Path],
) -> list[tuple[Path, Path]]:
    """Pair image and pose files by matching stems."""
    mat_by_stem = {Path(mat_path).stem: Path(mat_path) for mat_path in mat_paths}
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(Path(path) for path in image_paths):
        mat_path = mat_by_stem.get(image_path.stem)
        if mat_path is None:
            # TODO: Decide whether unmatched images should be fatal for all runs.
            continue
        pairs.append((image_path, mat_path))
    return pairs


def normalize_landmark_dataframe(poses_df: pd.DataFrame) -> pd.DataFrame:
    """Center landmark coordinates on the nose and scale them as in the notebooks."""
    features = [
        "forehead_",
        "nose_",
        "mouth_left_",
        "mouth_right_",
        "left_eye_",
        "chin_",
        "right_eye_",
    ]
    for dim in ("x", "y"):
        for feature in features:
            column = f"{feature}{dim}"
            if column not in poses_df.columns:
                raise ValueError(f"Missing column: {column}")

    normalized_df = poses_df.copy()
    for dim in ("x", "y"):
        nose_col = f"nose_{dim}"
        for feature in features:
            normalized_df[f"{feature}{dim}"] = poses_df[f"{feature}{dim}"] - poses_df[nose_col]

        diff = normalized_df[f"mouth_right_{dim}"] - normalized_df[f"left_eye_{dim}"]
        if (diff == 0).any():
            raise ZeroDivisionError("Scaling factor resulted in division by zero for some entries.")

        for feature in features:
            normalized_df[f"{feature}{dim}"] = normalized_df[f"{feature}{dim}"] / diff

    return normalized_df


def detect_face_landmarks(img: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Detect FaceMesh landmarks and return nose coordinates plus an annotated image."""
    with init_face_mesh() as face_mesh:
        result = face_mesh.process(img)

    img_h, img_w, _ = img.shape
    annotated_img = img.copy()
    nose_x, nose_y = None, None
    landmark_specs = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
    mesh_specs = mp.solutions.drawing_utils.DrawingSpec(thickness=1, color=(0, 255, 0))

    if result.multi_face_landmarks is not None:
        for face_landmarks in result.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_img,
                landmark_list=face_landmarks,
                landmark_drawing_spec=landmark_specs,
            )
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_img,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mesh_specs,
            )
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx == FEATURE_INDICES["NOSE"]:
                    nose_x = int(landmark.x * img_w)
                    nose_y = int(landmark.y * img_h)

    if nose_x is None or nose_y is None:
        raise ValueError("No nose landmark detected.")
    return nose_x, nose_y, annotated_img


def draw_axes(
    img: np.ndarray,
    pitch: float,
    yaw: float,
    roll: float,
    tx: int,
    ty: int,
    size: int = 50,
) -> np.ndarray:
    """Draw 3D head-pose axes on an image at the provided origin."""
    adjusted_yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, adjusted_yaw, roll]))[0].astype(np.float64)
    axes_points = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float64,
    )
    rotated_axes = rotation_matrix @ axes_points
    rotated_axes = (rotated_axes[:2, :] * size).astype(int)
    rotated_axes[0, :] += tx
    rotated_axes[1, :] += ty

    axes_properties = [
        {"end_point": (rotated_axes[0, 0], rotated_axes[1, 0]), "color": (255, 0, 0), "label": "X"},
        {"end_point": (rotated_axes[0, 1], rotated_axes[1, 1]), "color": (0, 255, 0), "label": "Y"},
        {"end_point": (rotated_axes[0, 2], rotated_axes[1, 2]), "color": (0, 0, 255), "label": "Z"},
    ]
    image_with_axes = img.copy()
    origin = (rotated_axes[0, 3], rotated_axes[1, 3])
    for axis in axes_properties:
        cv2.line(image_with_axes, origin, axis["end_point"], axis["color"], 3)
        cv2.putText(
            image_with_axes,
            axis["label"],
            axis["end_point"],
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            axis["color"],
            1,
        )
    return image_with_axes


def annotate_features(
    img: np.ndarray,
    features: Sequence[float | None],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Draw extracted normalized FaceMesh feature points on an image."""
    annotated_img = img.copy()
    for idx in range(len(features) // 2):
        x_value = features[idx * 2]
        y_value = features[idx * 2 + 1]
        if x_value is not None and y_value is not None:
            cv2.circle(
                annotated_img,
                center=(int(x_value * img_w), int(y_value * img_h)),
                radius=2,
                color=(255, 0, 0),
                thickness=2,
            )
    return annotated_img


def evaluate_pose_predictions(y_true: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute the MSE, R2, MAE, and MAPE metrics used in the training notebook."""
    metrics = {
        "Metric": ["MSE", "R2", "MAE", "MAPE (%)"],
        "Value": [
            mean_squared_error(y_true, y_pred),
            r2_score(y_true, y_pred),
            mean_absolute_error(y_true, y_pred),
            np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        ],
    }
    return pd.DataFrame(metrics)


def save_model(model: object, model_path: str | Path) -> None:
    """Persist a trained model with joblib."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(model_path: str | Path) -> object:
    """Load a trained model from disk."""
    return joblib.load(str(model_path))
