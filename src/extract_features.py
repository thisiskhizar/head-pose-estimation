"""Feature extraction and AFLW2000 pose-angle dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import pandas as pd

from src.utils import (
    FEATURE_INDICES,
    generate_feature_column_names,
    init_face_mesh,
    load_pose_angles,
    load_rgb_image,
    pair_image_and_mat_paths,
)


def extract_features(
    img: object,
    face_mesh: object,
    selected_points: Mapping[str, int] = FEATURE_INDICES,
    pose_angles: Sequence[float] | None = None,
) -> list[float | None]:
    """Extract selected MediaPipe FaceMesh landmarks and optional pose angles."""
    result = face_mesh.process(img)
    face_features: list[float | None] = []

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in selected_points.values():
                    face_features.append(float(landmark.x))
                    face_features.append(float(landmark.y))
    else:
        face_features.extend([None] * (len(selected_points) * 2))

    if pose_angles is not None:
        face_features.extend(float(angle) for angle in pose_angles)

    return face_features


def extract_features_from_path(
    image_path: str | Path,
    face_mesh: object,
    selected_points: Mapping[str, int] = FEATURE_INDICES,
    mat_path: str | Path | None = None,
) -> list[float | None]:
    """Load an image and extract model features, including pose angles when provided."""
    image_rgb, _, _ = load_rgb_image(image_path)
    pose_angles = load_pose_angles(mat_path) if mat_path is not None else None
    return extract_features(
        image_rgb,
        face_mesh=face_mesh,
        selected_points=selected_points,
        pose_angles=pose_angles,
    )


def extract_dataset_features(
    image_paths: Sequence[str | Path],
    mat_paths: Sequence[str | Path],
    face_mesh: object | None = None,
) -> pd.DataFrame:
    """Extract landmark features and pose labels for matched AFLW2000 files."""
    columns = generate_feature_column_names(include_pose=True)
    owns_face_mesh = face_mesh is None
    if face_mesh is None:
        face_mesh = init_face_mesh()

    rows: list[list[float | None]] = []
    try:
        for image_idx, (image_path, mat_path) in enumerate(
            pair_image_and_mat_paths(image_paths, mat_paths)
        ):
            try:
                row = extract_features_from_path(image_path, face_mesh, mat_path=mat_path)
            except Exception as exc:
                print(f"Error processing image {image_path}: {exc}")
                row = [None] * len(columns)
            rows.append(row)
            if image_idx % 100 == 0:
                print(f"Extracted features from {image_idx} images.")
    finally:
        if owns_face_mesh:
            face_mesh.close()

    return pd.DataFrame(rows, columns=columns)


def save_features_to_csv(
    images_paths: Sequence[str | Path],
    img_info_paths: Sequence[str | Path],
    csv_path: str | Path = "./data/processed/features_pose_angles.csv",
    face_mesh: object | None = None,
) -> pd.DataFrame:
    """Extract features and pose angles for all images, then save them to CSV."""
    poses_df = extract_dataset_features(images_paths, img_info_paths, face_mesh=face_mesh)
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    poses_df.to_csv(output_path, index=False)
    print(f"Feature data saved to {output_path}.")
    return poses_df


def build_dataset_csv(
    dataset_dir: str | Path = "./data/raw/AFLW2000",
    csv_path: str | Path = "./data/processed/features_pose_angles.csv",
) -> pd.DataFrame:
    """Discover AFLW2000 images and `.mat` files in a directory and write features."""
    dataset_dir = Path(dataset_dir)
    image_paths = sorted(dataset_dir.glob("*.jpg"))
    mat_paths = sorted(dataset_dir.glob("*.mat"))
    with init_face_mesh() as face_mesh:
        return save_features_to_csv(image_paths, mat_paths, csv_path=csv_path, face_mesh=face_mesh)


def extract_video_frame_features(
    frame_bgr: object,
    face_mesh: object,
    selected_points: Mapping[str, int] = FEATURE_INDICES,
) -> list[float | None]:
    """Convert a BGR video frame to RGB and extract the configured landmark features."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return extract_features(frame_rgb, face_mesh=face_mesh, selected_points=selected_points)
