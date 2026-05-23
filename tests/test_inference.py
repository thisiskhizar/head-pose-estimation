"""Image and video inference utilities for the trained head-pose model."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from src.extract_features import extract_features
from src.utils import (
    FEATURE_INDICES,
    detect_face_landmarks,
    draw_axes,
    generate_feature_column_names,
    init_face_mesh,
    load_model,
    load_rgb_image,
    normalize_landmark_dataframe,
)


def features_to_model_frame(features: list[float | None]) -> pd.DataFrame:
    """Convert extracted landmark features into one normalized model input row."""
    feature_columns = generate_feature_column_names(include_pose=False)
    features_df = pd.DataFrame([features], columns=feature_columns)
    return normalize_landmark_dataframe(features_df)


def predict_pose_from_rgb_image(img_rgb: object, model: object, face_mesh: object) -> tuple[float, float, float]:
    """Predict pitch, yaw, and roll from an RGB image array."""
    features = extract_features(img_rgb, face_mesh=face_mesh, selected_points=FEATURE_INDICES)
    if None in features:
        raise ValueError("No face detected.")
    normalized_features = features_to_model_frame(features)
    pitch, yaw, roll = model.predict(normalized_features)[0]
    return float(pitch), float(yaw), float(roll)


def infer_image(
    image_path: str | Path = "./archive/test-img.jpeg",
    model_path: str | Path = "./archive/model/m_hpe.pkl",
    output_path: str | Path | None = None,
    axes_size: int = 50,
) -> tuple[tuple[float, float, float], object]:
    """Run inference on an image and return predicted angles plus an annotated image."""
    model = load_model(model_path)
    img_rgb, _, _ = load_rgb_image(image_path)
    with init_face_mesh() as face_mesh:
        pitch, yaw, roll = predict_pose_from_rgb_image(img_rgb, model, face_mesh)
    nose_x, nose_y, _ = detect_face_landmarks(img_rgb)
    image_with_axes = draw_axes(img_rgb, pitch, yaw, roll, nose_x, nose_y, size=axes_size)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output = cv2.cvtColor(image_with_axes, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), output)

    return (pitch, yaw, roll), image_with_axes


def infer_video(
    video_source: str | int = 0,
    model_path: str | Path = "./archive/model/m_hpe.pkl",
    output_path: str | Path | None = None,
    display: bool = True,
) -> None:
    """Run head-pose inference on webcam or video frames."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    capture_source = video_source
    if isinstance(video_source, str):
        source_path = Path(video_source)
        if not source_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {source_path}. Replace 'input.mp4' with an existing video path."
            )
        capture_source = str(source_path)

    model = load_model(model_path)
    face_mesh = init_face_mesh()
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    writer = None
    frames_processed = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if frames_processed == 0:
                    raise RuntimeError(
                        f"No frames could be read from video source: {video_source}"
                    )
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                pitch, yaw, roll = predict_pose_from_rgb_image(frame_rgb, model, face_mesh)
                nose_x, nose_y, _ = detect_face_landmarks(frame_rgb)
                frame_with_axes = draw_axes(frame, pitch, yaw, roll, nose_x, nose_y)
            except ValueError:
                frame_with_axes = frame
                cv2.putText(
                    frame_with_axes,
                    "No face detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            if output_path is not None:
                writer = initialize_video_writer(writer, output_path, frame_with_axes)
                writer.write(frame_with_axes)
            frames_processed += 1

            if display:
                cv2.imshow("Head Pose Estimation", frame_with_axes)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        face_mesh.close()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def initialize_video_writer(writer: object, output_path: str | Path, frame: object) -> object:
    """Create a video writer lazily with frame dimensions from the first output frame."""
    if writer is not None:
        return writer
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, 20.0, (width, height))
