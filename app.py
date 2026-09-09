"""
app.py
------
Stage 5 of the MLOps workflow: Model Deployment (real-time REST inference).

In production this container would be built by CI/CD (Docker + GitHub Actions),
pushed to a registry (ECR/GCR), and deployed behind a load balancer on
Kubernetes / SageMaker / Vertex AI endpoints with autoscaling. A canary or
shadow-traffic strategy is used before routing 100% of traffic to a new model
version.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-risk-api")

app = FastAPI(title="Credit Risk Default Prediction API", version="1.0.0")

MODEL_URI = "models:/credit_risk_xgb/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)


class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    employment_length_years: float = Field(..., ge=0)
    loan_amount: float = Field(..., gt=0)
    loan_term_months: int = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=900)
    existing_loans_count: int = Field(..., ge=0)
    debt_to_income_ratio: float = Field(..., ge=0)
    num_late_payments_12m: int = Field(..., ge=0)
    home_ownership: str
    purpose: str


class PredictionResponse(BaseModel):
    default_probability: float
    risk_band: str
    model_version: str
    latency_ms: float


def band_from_probability(p: float) -> str:
    if p < 0.10:
        return "LOW"
    elif p < 0.30:
        return "MEDIUM"
    else:
        return "HIGH"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    start = time.time()
    try:
        raw = pd.DataFrame([application.dict()])

        # NOTE: reuse the exact same feature engineering used in training
        # to avoid train/serve skew (imported from data_pipeline.engineer_features
        # in a real deployment; inlined-conceptually here).
        from data_pipeline import engineer_features
        features = engineer_features(raw)

        proba = float(model.predict(features)[0])
        latency_ms = (time.time() - start) * 1000

        logger.info(
            "prediction served",
            extra={"proba": proba, "latency_ms": latency_ms},
        )

        return PredictionResponse(
            default_probability=round(proba, 4),
            risk_band=band_from_probability(proba),
            model_version="Production",
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
