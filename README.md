# Credit Risk / Loan Default Prediction — MLOps Pipeline

An end-to-end MLOps workflow for predicting the probability that a loan
applicant will default, covering data ingestion through monitoring and
retraining.

## Architecture overview

```
<img width="3024" height="1206" alt="Architecture overview" src="https://github.com/user-attachments/assets/affd0111-ea91-4dae-839a-641d1da41ac6" />
```

## Repo contents

| File | Pipeline stage | Purpose |
|---|---|---|
| `data_pipeline.py` | Data ingestion & validation, feature engineering | Schema checks, null/range validation, deterministic feature transforms shared by training and serving |
| `train_model.py` | Model training, experiment tracking, model registry | Trains XGBoost, logs params/metrics/artifacts to MLflow, auto-promotes to Production if AUC clears threshold |
| `app.py` | Deployment | FastAPI REST endpoint (`/predict`, `/health`) serving the Production model |
| `monitor.py` | Monitoring & maintenance | KS-test feature/prediction drift detection, live AUC decay check, retraining trigger |
| `requirements.txt` | — | Pinned Python dependencies |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Start a local MLflow tracking server (or point MLFLOW_TRACKING_URI at a
# remote one in production)
mlflow server --host 0.0.0.0 --port 5000
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Expected raw data at `data/raw/loan_applications.csv` with columns:
`customer_id, age, annual_income, employment_length_years, loan_amount,
loan_term_months, credit_score, existing_loans_count, debt_to_income_ratio,
num_late_payments_12m, home_ownership, purpose, default_flag`.

## Running each stage

**1. Data pipeline**
```bash
python data_pipeline.py
```
Validates the raw CSV and writes engineered features to
`data/processed/loan_features.parquet`.

**2. Train + register the model**
```bash
python train_model.py
```
Trains XGBoost, logs the run to MLflow, and promotes the model to
`Production` in the registry if ROC-AUC ≥ 0.75 (adjust `min_auc` in
`promote_if_better`).

**3. Serve predictions**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```
Then:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 34, "annual_income": 65000, "employment_length_years": 5,
    "loan_amount": 15000, "loan_term_months": 36, "credit_score": 690,
    "existing_loans_count": 1, "debt_to_income_ratio": 0.32,
    "num_late_payments_12m": 0, "home_ownership": "RENT",
    "purpose": "debt_consolidation"
  }'
```

**4. Monitor for drift / decay**
```bash
python monitor.py
```
Compares a current production batch (`data/processed/loan_features_latest_batch.parquet`)
against the training reference set and prints/alerts if retraining is
recommended.

## Production hardening still needed

This repo is a working skeleton for coursework/demo purposes. A real
deployment would add:

- **Orchestration** — an Airflow/Prefect DAG chaining `data_pipeline.py` →
  `train_model.py` → `monitor.py` on a schedule, with retries and alerting.
- **CI/CD** — GitHub Actions (or similar) to lint/test code, build a Docker
  image for `app.py`, and deploy to Kubernetes/SageMaker/Vertex AI with a
  canary or shadow-traffic rollout before full cutover.
- **Feature store** — a shared store (e.g. Feast) so training and serving
  read identical feature definitions instead of duplicating logic.
- **Fairness/bias checks** — since this is a lending decision, the
  promotion gate in `train_model.py` should also check disparate-impact
  metrics across protected groups (e.g. age, gender where legally
  permissible to evaluate) before promoting to Production.
- **Explainability** — SHAP values logged per prediction for adverse-action
  notices and regulatory compliance (fair lending laws typically require
  reasons for a denial).
- **Secrets/config management** — MLflow URIs, DB credentials, etc. via a
  secrets manager rather than hardcoded values.
- **Human-in-the-loop review** — a dashboard for risk analysts to review
  borderline/high-risk predictions before final approval, especially early
  after a new model version ships.
