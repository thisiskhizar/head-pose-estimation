"""Model training pipeline for head-pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from src.utils import POSE_COLUMNS, evaluate_pose_predictions, normalize_landmark_dataframe, save_model


@dataclass
class TrainingResult:
    """Container for trained model artifacts and evaluation outputs."""

    model: RandomForestRegressor
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    y_test_pred: object
    X_test: pd.DataFrame
    y_test: pd.DataFrame


def load_and_clean_dataset(
    data_path: str | Path = "./data/processed/features_pose_angles.csv",
) -> pd.DataFrame:
    """Load extracted pose features, remove duplicates, and drop missing rows."""
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    missing_values = df.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    return df.dropna()


def split_features_targets(
    normalized_df: pd.DataFrame,
    test_size: float = 0.3,
    validation_fraction_of_temp: float = 0.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split normalized data into train, validation, and test sets."""
    X = normalized_df.drop(list(POSE_COLUMNS), axis=1)
    y = normalized_df[list(POSE_COLUMNS)]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_fraction_of_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    """Create the Random Forest regressor used by the notebooks."""
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)


def tune_model(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv: int = 3,
) -> RandomForestRegressor:
    """Run the notebook grid search and return the best estimator."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def train_pipeline(
    data_path: str | Path = "./data/processed/features_pose_angles.csv",
    model_path: str | Path = "./models/m_hpe.pkl",
    use_grid_search: bool = True,
) -> TrainingResult:
    """Run the full cleaning, normalization, split, train, save, and evaluate pipeline."""
    df = load_and_clean_dataset(data_path)
    normalized_df = normalize_landmark_dataframe(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_features_targets(normalized_df)

    model = build_model()
    if use_grid_search:
        model = tune_model(model, X_train, y_train)
    model.fit(X_train, y_train)
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    validation_metrics = evaluate_pose_predictions(y_val, y_val_pred)
    test_metrics = evaluate_pose_predictions(y_test, y_test_pred)
    return TrainingResult(
        model=model,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        y_test_pred=y_test_pred,
        X_test=X_test,
        y_test=y_test,
    )


def calculate_residuals(y_true: pd.DataFrame, y_pred: object) -> pd.DataFrame:
    """Calculate residuals between true and predicted pose angles."""
    return y_true - y_pred


def calculate_mape_per_angle(y_true: pd.DataFrame, y_pred: object) -> pd.Series:
    """Calculate mean absolute percentage error separately for pitch, yaw, and roll."""
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true), axis=0)


def plot_residuals(y_true: pd.DataFrame, y_pred: object) -> None:
    """Plot the residual distribution used in the training notebook."""
    residuals = calculate_residuals(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color="purple", bins=30)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()


def plot_actual_vs_predicted(y_true: pd.DataFrame, y_pred: object) -> None:
    """Plot actual pose values against predicted pose values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color="blue")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.title("Actual vs. Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()


def plot_mape_per_angle(y_true: pd.DataFrame, y_pred: object) -> None:
    """Plot MAPE for pitch, yaw, and roll."""
    angles = list(POSE_COLUMNS)
    mape_per_angle = calculate_mape_per_angle(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=angles, y=mape_per_angle, palette="viridis")
    plt.title("MAPE for Each Angle")
    plt.xlabel("Head Pose Angle")
    plt.ylabel("MAPE (%)")
    plt.show()
