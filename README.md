# Head Pose Estimation

Head pose estimation using MediaPipe facial landmarks to predict pitch, yaw, and roll angles. The project includes feature extraction from the AFLW2000 dataset, model training, exploratory analysis, and inference on images or videos.

## Project Structure

```text
data/
  raw/                     # Original dataset files
  processed/
    features_pose_angles.csv
src/
  extract_features.py      # Feature extraction and pose-angle CSV creation
  train.py                 # Training, tuning, evaluation, and model saving
  utils.py                 # Shared image, model, normalization, and drawing helpers
notebooks/
  01_eda.ipynb             # Exploratory analysis and visualization only
tests/
  test_inference.py        # Image and video inference helpers
requirements.txt           # Pinned project dependencies
```

## 1. Create And Activate An Environment

From the project root:

```bash
python -m venv venv
```

On Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Prepare The Dataset

Place the AFLW2000 images and `.mat` annotation files in:

```text
data/raw/AFLW2000/
```

Each image should have a matching `.mat` file with the same stem, for example:

```text
image00002.jpg
image00002.mat
```

## 4. Extract Features

Generate the feature CSV used for training:

```bash
python -c "from src.extract_features import build_dataset_csv; build_dataset_csv('data/raw/AFLW2000', 'data/processed/features_pose_angles.csv')"
```

This creates:

```text
data/processed/features_pose_angles.csv
```

The CSV contains selected FaceMesh landmark coordinates plus `pitch`, `yaw`, and `roll`.

## 5. Train The Model

Run the full training pipeline:

```bash
python -c "from src.train import train_pipeline; result = train_pipeline('data/processed/features_pose_angles.csv', 'models/m_hpe.pkl'); print(result.validation_metrics); print(result.test_metrics)"
```

This performs cleaning, normalization, train/validation/test splitting, Random Forest grid search, model fitting, saving, and evaluation.

The trained model is saved to:

```text
models/m_hpe.pkl
```

## 6. Run Exploratory Analysis

Open:

```text
notebooks/01_eda.ipynb
```

Use this notebook for dataset counts, sample image viewing, FaceMesh visualization, pose-axis visualization, selected feature-point visualization, and `data/processed/features_pose_angles.csv` summary checks.

## 7. Run Image Inference

Use the inference helper on a single image:

```bash
python -c "from tests.test_inference import infer_image; angles, _ = infer_image('archive/test-img.jpeg', 'models/m_hpe.pkl', 'outputs/test-img-axes.jpg'); print(angles)"
```

The returned tuple is:

```text
(pitch, yaw, roll)
```

If `output_path` is provided, an annotated image with pose axes is written there.

## 8. Run Video Or Webcam Inference

Use webcam input:

```bash
python -c "from tests.test_inference import infer_video; infer_video(0, 'models/m_hpe.pkl')"
```

Use a video file:

```bash
python -c "from tests.test_inference import infer_video; infer_video('input.mp4', 'models/m_hpe.pkl', 'outputs/head-pose.mp4', display=False)"
```

Press `q` to stop webcam display mode.

## 9. Verify Syntax

```bash
python -m compileall src tests
```

## Notes

- Run commands from the repository root so imports like `src.train` resolve correctly.
- The feature extraction step uses MediaPipe FaceMesh and can take time on the full AFLW2000 dataset.
- If a dataset image has no detected face or a missing annotation pair, the code preserves the row-handling behavior from the notebooks and leaves a TODO where policy decisions are unclear.
