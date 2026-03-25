"""
api/main.py
-----------
FastAPI backend for Loan Default Prediction.
Endpoint: POST /predict
- Validates input with Pydantic
- Scores loan application with XGBoost
- Returns default probability + SHAP explanation + risk level
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Global artifact store ─────────────────────────────────────────────────────
_artifacts: dict = {}

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# All 45 feature names in exact training order
FEATURE_NAMES = [
    "term", "int_rate", "emp_length", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "total_rec_prncp",
    "total_rec_int", "last_pymnt_amnt",
    "earliest_cr_line_M", "earliest_cr_line_Y",
    "issue_d_M", "issue_d_Y",
    "last_pymnt_d_M", "last_pymnt_d_Y",
    "last_credit_pull_d_M", "last_credit_pull_d_Y",
    "home_ownership_NONE", "home_ownership_OTHER",
    "home_ownership_OWN", "home_ownership_RENT",
    "verification_status_Source Verified", "verification_status_Verified",
    "purpose_credit_card", "purpose_debt_consolidation",
    "purpose_educational", "purpose_home_improvement",
    "purpose_house", "purpose_major_purchase", "purpose_medical",
    "purpose_moving", "purpose_other", "purpose_renewable_energy",
    "purpose_small_business", "purpose_vacation", "purpose_wedding",
    "debt_to_income", "installment_to_income", "repayment_ratio"
]


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts from %s ...", MODEL_DIR)
    try:
        _artifacts["model"]     = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
        _artifacts["scaler"]    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        _artifacts["explainer"] = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
        _artifacts["threshold"] = joblib.load(os.path.join(MODEL_DIR, "optimal_threshold.pkl"))
        _artifacts["features"]  = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        logger.info("All artifacts loaded. Threshold = %.2f", _artifacts["threshold"])
    except Exception as e:
        logger.error("Failed to load artifacts: %s", e)
        raise
    yield
    _artifacts.clear()
    logger.info("Artifacts cleared on shutdown.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts probability of loan default using XGBoost + SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ────────────────────────────────────────────────────────────
class LoanInput(BaseModel):
    """
    User-facing fields only — the API derives all 45 model features internally.
    This makes the frontend simple while keeping the model's full feature set.
    """
    # Core numeric features
    term:             int   = Field(..., ge=36, le=60,      description="Loan term in months (36 or 60)")
    int_rate:         float = Field(..., ge=0,  le=35,      description="Interest rate (%)")
    emp_length:       float = Field(..., ge=0,  le=11,      description="Employment length (0-11, use 0.5 for < 1 year)")
    annual_inc:       float = Field(..., ge=0,              description="Annual income ($)")
    dti:              float = Field(..., ge=0,  le=100,     description="Debt-to-income ratio")
    delinq_2yrs:      float = Field(0,  ge=0,              description="Delinquencies in last 2 years")
    inq_last_6mths:   float = Field(0,  ge=0,              description="Credit inquiries in last 6 months")
    open_acc:         float = Field(..., ge=0,              description="Number of open credit lines")
    pub_rec:          float = Field(0,  ge=0,              description="Public derogatory records")
    revol_bal:        float = Field(..., ge=0,              description="Total revolving balance ($)")
    revol_util:       float = Field(..., ge=0,  le=100,    description="Revolving line utilisation (%)")
    total_acc:        float = Field(..., ge=0,              description="Total number of credit lines")
    total_rec_prncp:  float = Field(..., ge=0,              description="Principal received to date ($)")
    total_rec_int:    float = Field(..., ge=0,              description="Interest received to date ($)")
    last_pymnt_amnt:  float = Field(..., ge=0,              description="Last payment amount ($)")
    loan_amnt:        float = Field(..., ge=0,              description="Loan amount ($) — used to compute ratios")
    installment:      float = Field(..., ge=0,              description="Monthly installment ($) — used to compute ratios")
    total_pymnt:      float = Field(..., ge=0,              description="Total payment received ($) — used to compute ratios")

    # Date features (month and year)
    earliest_cr_line_M:    int = Field(..., ge=1, le=12)
    earliest_cr_line_Y:    int = Field(..., ge=1950, le=2025)
    issue_d_M:             int = Field(..., ge=1, le=12)
    issue_d_Y:             int = Field(..., ge=2000, le=2025)
    last_pymnt_d_M:        int = Field(..., ge=1, le=12)
    last_pymnt_d_Y:        int = Field(..., ge=2000, le=2025)
    last_credit_pull_d_M:  int = Field(..., ge=1, le=12)
    last_credit_pull_d_Y:  int = Field(..., ge=2000, le=2025)

    # Categorical fields (raw strings — API encodes internally)
    home_ownership:       str = Field(..., description="RENT, MORTGAGE, OWN, OTHER, NONE")
    verification_status:  str = Field(..., description="Not Verified, Source Verified, Verified")
    purpose:              str = Field(..., description="debt_consolidation, credit_card, home_improvement, etc.")

    @validator("home_ownership")
    def validate_home(cls, v):
        valid = {"RENT", "MORTGAGE", "OWN", "OTHER", "NONE"}
        if v.upper() not in valid:
            raise ValueError(f"home_ownership must be one of {valid}")
        return v.upper()

    @validator("purpose")
    def validate_purpose(cls, v):
        valid = {"debt_consolidation","credit_card","home_improvement","major_purchase",
                 "small_business","other","medical","car","moving","vacation",
                 "house","educational","wedding","renewable_energy"}
        if v.lower() not in valid:
            raise ValueError(f"purpose must be one of {valid}")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "term": 36, "int_rate": 13.5, "emp_length": 5.0,
                "annual_inc": 55000, "dti": 18.5, "delinq_2yrs": 0,
                "inq_last_6mths": 1, "open_acc": 9, "pub_rec": 0,
                "revol_bal": 8400, "revol_util": 67.5, "total_acc": 15,
                "total_rec_prncp": 1200, "total_rec_int": 280,
                "last_pymnt_amnt": 350, "loan_amnt": 10000,
                "installment": 340, "total_pymnt": 1480,
                "earliest_cr_line_M": 6, "earliest_cr_line_Y": 2005,
                "issue_d_M": 3, "issue_d_Y": 2013,
                "last_pymnt_d_M": 8, "last_pymnt_d_Y": 2014,
                "last_credit_pull_d_M": 9, "last_credit_pull_d_Y": 2014,
                "home_ownership": "RENT",
                "verification_status": "Verified",
                "purpose": "debt_consolidation"
            }
        }


class PredictionResponse(BaseModel):
    default_probability: float  = Field(..., description="Probability of default (0–1)")
    prediction_label:    str    = Field(..., description="DEFAULT or NON-DEFAULT")
    risk_level:          str    = Field(..., description="LOW, MEDIUM, or HIGH")
    threshold_used:      float  = Field(..., description="Classification threshold applied")
    expected_loss_usd:   float  = Field(..., description="Expected loss if loan approved ($)")
    top_shap_features:   Dict[str, float] = Field(..., description="Top 10 SHAP feature contributions")


# ── Helper — build feature vector ────────────────────────────────────────────
def build_feature_vector(data: LoanInput, scaler, feature_names: list) -> np.ndarray:
    """
    Converts user-facing LoanInput into the 45-feature vector the model expects.
    Handles one-hot encoding and ratio engineering internally.
    """
    # Base numeric features
    row = {
        "term":            data.term,
        "int_rate":        data.int_rate,
        "emp_length":      data.emp_length,
        "annual_inc":      data.annual_inc,
        "dti":             data.dti,
        "delinq_2yrs":     data.delinq_2yrs,
        "inq_last_6mths":  data.inq_last_6mths,
        "open_acc":        data.open_acc,
        "pub_rec":         data.pub_rec,
        "revol_bal":       data.revol_bal,
        "revol_util":      data.revol_util,
        "total_acc":       data.total_acc,
        "total_rec_prncp": data.total_rec_prncp,
        "total_rec_int":   data.total_rec_int,
        "last_pymnt_amnt": data.last_pymnt_amnt,
        # Date features
        "earliest_cr_line_M":   data.earliest_cr_line_M,
        "earliest_cr_line_Y":   data.earliest_cr_line_Y,
        "issue_d_M":            data.issue_d_M,
        "issue_d_Y":            data.issue_d_Y,
        "last_pymnt_d_M":       data.last_pymnt_d_M,
        "last_pymnt_d_Y":       data.last_pymnt_d_Y,
        "last_credit_pull_d_M": data.last_credit_pull_d_M,
        "last_credit_pull_d_Y": data.last_credit_pull_d_Y,
        # Engineered ratios
        "debt_to_income":        data.loan_amnt / (data.annual_inc + 1),
        "installment_to_income": data.installment / ((data.annual_inc / 12) + 1),
        "repayment_ratio":       data.total_pymnt / (data.loan_amnt + 1),
    }

    # One-hot encoding — home_ownership (drop_first removes MORTGAGE)
    row["home_ownership_NONE"]  = 1.0 if data.home_ownership == "NONE"     else 0.0
    row["home_ownership_OTHER"] = 1.0 if data.home_ownership == "OTHER"    else 0.0
    row["home_ownership_OWN"]   = 1.0 if data.home_ownership == "OWN"      else 0.0
    row["home_ownership_RENT"]  = 1.0 if data.home_ownership == "RENT"     else 0.0

    # verification_status (drop_first removes Not Verified)
    row["verification_status_Source Verified"] = 1.0 if data.verification_status == "Source Verified" else 0.0
    row["verification_status_Verified"]        = 1.0 if data.verification_status == "Verified"        else 0.0

    # purpose (drop_first removes car)
    for p in ["credit_card","debt_consolidation","educational","home_improvement",
              "house","major_purchase","medical","moving","other",
              "renewable_energy","small_business","vacation","wedding"]:
        row[f"purpose_{p}"] = 1.0 if data.purpose == p else 0.0

    # Build DataFrame in exact feature order
    df = pd.DataFrame([row])[feature_names]

    # Scale numeric columns using the saved scaler
    numeric_cols = [c for c in feature_names if not any(
        c.startswith(p) for p in
        ["home_ownership_","verification_status_","purpose_"]
    )]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df.values, feature_names


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Loan Default Prediction API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    loaded = "model" in _artifacts
    return {
        "status":       "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "threshold":    _artifacts.get("threshold", None)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(loan: LoanInput):
    """
    Score a loan application for default risk.
    Returns probability, label, risk level, expected loss, and SHAP explanations.
    """
    if "model" not in _artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        X, feat_names = build_feature_vector(
            loan,
            _artifacts["scaler"],
            _artifacts["features"]
        )

        model     = _artifacts["model"]
        threshold = _artifacts["threshold"]
        explainer = _artifacts["explainer"]

        # Predict
        prob  = float(model.predict_proba(X)[0, 1])
        label = "DEFAULT" if prob >= threshold else "NON-DEFAULT"

        # Risk level
        if prob < 0.3:
            risk = "LOW"
        elif prob < 0.6:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        # Expected loss (probability × average loan loss)
        expected_loss = round(prob * 5000, 2)

        # SHAP explanation
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]

        contributions = dict(zip(feat_names, sv.tolist()))
        top_shap = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )
        top_shap = {k: round(v, 5) for k, v in top_shap.items()}

        return PredictionResponse(
            default_probability = round(prob, 4),
            prediction_label    = label,
            risk_level          = risk,
            threshold_used      = round(threshold, 2),
            expected_loss_usd   = expected_loss,
            top_shap_features   = top_shap
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", tags=["Meta"])
def model_info():
    if "model" not in _artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type":    type(_artifacts["model"]).__name__,
        "n_features":    len(_artifacts["features"]),
        "threshold":     _artifacts["threshold"],
        "feature_names": _artifacts["features"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)