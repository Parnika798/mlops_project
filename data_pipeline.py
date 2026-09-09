"""
data_pipeline.py
-----------------
Stage 1 & 2 of the MLOps workflow: Data Ingestion + Data Validation + Feature Engineering.

In production this module would run as a scheduled job (Airflow / Prefect / Databricks
Job) that reads from a data warehouse (e.g. Snowflake, BigQuery) or a feature store,
validates the incoming batch against expected schema/statistics, engineers features,
and writes a clean, versioned dataset to the feature store / training data lake.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------- #
# 1. Data schema / validation
# --------------------------------------------------------------------------- #

EXPECTED_COLUMNS = {
    "customer_id": "object",
    "age": "int64",
    "annual_income": "float64",
    "employment_length_years": "float64",
    "loan_amount": "float64",
    "loan_term_months": "int64",
    "credit_score": "int64",
    "existing_loans_count": "int64",
    "debt_to_income_ratio": "float64",
    "num_late_payments_12m": "int64",
    "home_ownership": "object",       # RENT / OWN / MORTGAGE
    "purpose": "object",              # e.g. debt_consolidation, education
    "default_flag": "int64",          # target: 1 = defaulted, 0 = paid on time
}


@dataclass
class ValidationReport:
    ok: bool
    missing_columns: list
    null_rate: dict
    out_of_range: dict


def validate_schema(df: pd.DataFrame) -> ValidationReport:
    """Basic data-quality gate that must pass before data enters the training pipeline.
    Mirrors what a tool like Great Expectations / TFDV would enforce automatically."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    null_rate = {c: float(df[c].isna().mean()) for c in df.columns if c in EXPECTED_COLUMNS}

    out_of_range = {}
    if "age" in df.columns:
        bad = df[(df["age"] < 18) | (df["age"] > 100)]
        if len(bad):
            out_of_range["age"] = len(bad)
    if "credit_score" in df.columns:
        bad = df[(df["credit_score"] < 300) | (df["credit_score"] > 900)]
        if len(bad):
            out_of_range["credit_score"] = len(bad)
    if "debt_to_income_ratio" in df.columns:
        bad = df[(df["debt_to_income_ratio"] < 0) | (df["debt_to_income_ratio"] > 5)]
        if len(bad):
            out_of_range["debt_to_income_ratio"] = len(bad)

    high_null_cols = {c: r for c, r in null_rate.items() if r > 0.05}

    ok = (len(missing) == 0) and (len(high_null_cols) == 0) and (len(out_of_range) == 0)
    return ValidationReport(ok=ok, missing_columns=missing, null_rate=high_null_cols, out_of_range=out_of_range)


# --------------------------------------------------------------------------- #
# 2. Feature engineering
# --------------------------------------------------------------------------- #

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, versioned feature transformations. Keeping this pure and
    stateless (no fitted objects) means the exact same code can run in training
    and at inference time, avoiding train/serve skew."""

    df = df.copy()

    # Impute a small number of expected nulls with sensible defaults / medians
    df["employment_length_years"] = df["employment_length_years"].fillna(0)
    df["debt_to_income_ratio"] = df["debt_to_income_ratio"].fillna(
        df["debt_to_income_ratio"].median()
    )

    # Derived risk-relevant features
    df["loan_to_income_ratio"] = df["loan_amount"] / df["annual_income"].replace(0, np.nan)
    df["loan_to_income_ratio"] = df["loan_to_income_ratio"].fillna(df["loan_to_income_ratio"].median())

    df["monthly_installment_est"] = df["loan_amount"] / df["loan_term_months"]

    df["income_per_dependent_proxy"] = df["annual_income"] / (df["existing_loans_count"] + 1)

    df["is_high_risk_purpose"] = df["purpose"].isin(
        ["debt_consolidation", "medical", "small_business"]
    ).astype(int)

    df["recent_delinquency_flag"] = (df["num_late_payments_12m"] > 0).astype(int)

    # One-hot encode low-cardinality categoricals
    df = pd.get_dummies(df, columns=["home_ownership", "purpose"], drop_first=True)

    return df


# --------------------------------------------------------------------------- #
# 3. Orchestration entry point
# --------------------------------------------------------------------------- #

def run_pipeline(raw_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    report = validate_schema(df)
    if not report.ok:
        raise ValueError(f"Data validation failed: {report}")

    features_df = engineer_features(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    print(f"Validated + engineered dataset written to {output_path} "
          f"({features_df.shape[0]} rows, {features_df.shape[1]} cols)")
    return features_df


if __name__ == "__main__":
    run_pipeline(
        raw_path="data/raw/loan_applications.csv",
        output_path="data/processed/loan_features.parquet",
    )
