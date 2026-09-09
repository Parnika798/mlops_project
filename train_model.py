"""
train_model.py
---------------
Stage 3 & 4 of the MLOps workflow: Model Training + Experiment Tracking + Model Registry.

Run as a CI/CD-triggered job (e.g. GitHub Actions -> a training cluster, or a
Kubeflow/SageMaker pipeline step). Every run is logged to MLflow so results are
reproducible, comparable, and the best model can be promoted to the registry.
"""

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from data_pipeline import run_pipeline

TARGET_COL = "default_flag"
DROP_COLS = ["customer_id", TARGET_COL]

MLFLOW_EXPERIMENT = "credit-risk-default-prediction"
REGISTERED_MODEL_NAME = "credit_risk_xgb"


def load_training_data(processed_path: str) -> pd.DataFrame:
    return pd.read_parquet(processed_path)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_and_log(X_train, y_train, X_test, y_test, params: dict):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        # Class imbalance is typical in default prediction (defaults are rare)
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        params = {**params, "scale_pos_weight": scale_pos_weight}

        model = XGBClassifier(
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            **params,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm)
        print(classification_report(y_test, y_pred))

        # Log feature importance as an artifact for model-governance / explainability review
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        importances.sort_values(ascending=False).to_csv("feature_importance.csv")
        mlflow.log_artifact("feature_importance.csv")

        # Log + register the model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        joblib.dump(model, "model.joblib")
        mlflow.log_artifact("model.joblib")

        print(f"Run {run.info.run_id} logged. Metrics: {metrics}")
        return run.info.run_id, metrics


def promote_if_better(run_id: str, metrics: dict, min_auc: float = 0.75):
    """Simple automated gate: only promote to 'Production' in the registry if
    the model clears a minimum quality bar. In a real pipeline this would also
    check fairness metrics across protected groups before promotion."""
    client = mlflow.tracking.MlflowClient()

    if metrics["roc_auc"] < min_auc:
        print(f"Model AUC {metrics['roc_auc']:.3f} below threshold {min_auc}. "
              f"Not promoting — staying in 'Staging'.")
        return

    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        print("No registered model version found for this run.")
        return

    latest_version = versions[0].version
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted model version {latest_version} to Production.")


if __name__ == "__main__":
    df = run_pipeline(
        raw_path="data/raw/loan_applications.csv",
        output_path="data/processed/loan_features.parquet",
    )

    X_train, X_test, y_train, y_test = split_data(df)

    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    run_id, metrics = train_and_log(X_train, y_train, X_test, y_test, params)
    promote_if_better(run_id, metrics)
