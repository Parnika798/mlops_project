"""
monitor.py
----------
Stage 6 of the MLOps workflow: Monitoring & Maintenance.

Runs on a schedule (e.g. daily Airflow DAG) comparing the latest scored
production batch against the training reference dataset to catch:
  1. Data drift        (input feature distributions shifting)
  2. Prediction drift   (model output distribution shifting)
  3. Performance decay  (once ground-truth defaults are observed, weeks later)

Alerts feed into Slack/PagerDuty and can trigger an automated retraining job.
"""

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
import json


DRIFT_PVALUE_THRESHOLD = 0.05     # below this -> statistically significant drift
DRIFT_FEATURE_ALERT_RATIO = 0.3   # if >30% of features drift -> trigger retraining
MIN_ACCEPTABLE_AUC = 0.70


def detect_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                          numeric_cols: list) -> dict:
    """Kolmogorov-Smirnov test per numeric feature between training-time
    reference data and the current production window."""
    results = {}
    for col in numeric_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        stat, p_value = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
        results[col] = {"ks_stat": float(stat), "p_value": float(p_value),
                         "drifted": bool(p_value < DRIFT_PVALUE_THRESHOLD)}
    return results


def detect_prediction_drift(reference_scores: pd.Series, current_scores: pd.Series) -> dict:
    stat, p_value = ks_2samp(reference_scores.dropna(), current_scores.dropna())
    return {"ks_stat": float(stat), "p_value": float(p_value),
            "drifted": bool(p_value < DRIFT_PVALUE_THRESHOLD)}


def evaluate_live_performance(y_true: pd.Series, y_proba: pd.Series) -> dict:
    """Once actual repayment/default outcomes are known (with a lag),
    recompute real performance to catch silent model decay."""
    auc = roc_auc_score(y_true, y_proba)
    return {"live_auc": float(auc), "below_threshold": bool(auc < MIN_ACCEPTABLE_AUC)}


def build_monitoring_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                             numeric_cols: list, reference_scores: pd.Series,
                             current_scores: pd.Series,
                             live_labels: pd.Series = None,
                             live_scores: pd.Series = None) -> dict:

    feature_drift = detect_feature_drift(reference_df, current_df, numeric_cols)
    pred_drift = detect_prediction_drift(reference_scores, current_scores)

    drifted_feature_ratio = (
        sum(v["drifted"] for v in feature_drift.values()) / max(len(feature_drift), 1)
    )

    report = {
        "feature_drift": feature_drift,
        "drifted_feature_ratio": drifted_feature_ratio,
        "prediction_drift": pred_drift,
        "retrain_recommended": drifted_feature_ratio > DRIFT_FEATURE_ALERT_RATIO
        or pred_drift["drifted"],
    }

    if live_labels is not None and live_scores is not None:
        perf = evaluate_live_performance(live_labels, live_scores)
        report["live_performance"] = perf
        report["retrain_recommended"] = report["retrain_recommended"] or perf["below_threshold"]

    return report


def send_alert(report: dict):
    """Stub — wire this to Slack webhook / PagerDuty / email in production."""
    if report["retrain_recommended"]:
        print("ALERT: drift or performance decay detected -> triggering retraining pipeline")
        print(json.dumps(report, indent=2, default=str))
    else:
        print("Monitoring check passed. No action needed.")


if __name__ == "__main__":
    reference_df = pd.read_parquet("data/processed/loan_features.parquet")
    current_df = pd.read_parquet("data/processed/loan_features_latest_batch.parquet")

    numeric_cols = [
        "age", "annual_income", "loan_amount", "credit_score",
        "debt_to_income_ratio", "loan_to_income_ratio",
    ]

    report = build_monitoring_report(
        reference_df=reference_df,
        current_df=current_df,
        numeric_cols=numeric_cols,
        reference_scores=reference_df["default_flag"],   # placeholder scores
        current_scores=current_df["default_flag"],
    )
    send_alert(report)
