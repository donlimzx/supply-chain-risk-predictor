"""
Phase 3 — Supply Chain Risk Prediction API
==========================================
Endpoints:
    GET  /health         Liveness check
    GET  /metrics        Model performance stats
    POST /predict        Single shipment → risk_score + risk_tier
    POST /bulk_predict   List of shipments → same, in batch

Run:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Model loading ────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

app = FastAPI(
    title="Supply Chain Risk Predictor",
    description="Predicts late-delivery risk for supply chain shipments.",
    version="1.0.0",
)

_model_payload: dict[str, Any] = {}


@app.on_event("startup")
def load_model() -> None:
    global _model_payload
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    _model_payload = joblib.load(MODEL_PATH)


def _pipeline():
    return _model_payload["model"]


# ── Risk tier thresholds ──────────────────────────────────────────────────────
#   low  : score < 0.40
#   medium: 0.40 ≤ score < 0.70
#   high  : score ≥ 0.70

def _tier(score: float) -> str:
    if score < 0.40:
        return "low"
    if score < 0.70:
        return "medium"
    return "high"


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ShipmentInput(BaseModel):
    # ── Required fields (the three strongest predictors) ──────────────────
    shipping_mode: str = Field(
        ..., alias="Shipping Mode",
        description="One of: Standard Class | Second Class | First Class | Same Day",
    )
    days_for_shipment_scheduled: int = Field(
        ..., ge=0, le=30, alias="Days for shipment (scheduled)",
        description="Contracted SLA shipping days",
    )
    market: str = Field(
        ..., alias="Market",
        description="One of: LATAM | Europe | Pacific Asia | USCA | Africa",
    )

    # ── Optional categorical fields (dataset mode as default) ─────────────
    order_region: str = Field("Central America", alias="Order Region")
    customer_segment: str = Field("Consumer", alias="Customer Segment")
    department_name: str = Field("Fan Shop", alias="Department Name")
    category_name: str = Field("Cleats", alias="Category Name")
    payment_type: str = Field("DEBIT", alias="Type")

    # ── Optional numeric fields (dataset median as default) ───────────────
    benefit_per_order: float = Field(31.52, alias="Benefit per order")
    sales_per_customer: float = Field(163.99, alias="Sales per customer")
    order_item_discount: float = Field(14.0, alias="Order Item Discount")
    order_item_discount_rate: float = Field(0.10, alias="Order Item Discount Rate")
    order_item_product_price: float = Field(59.99, alias="Order Item Product Price")
    order_item_profit_ratio: float = Field(0.27, alias="Order Item Profit Ratio")
    order_item_quantity: int = Field(1, ge=1, alias="Order Item Quantity")
    sales: float = Field(199.92, alias="Sales")
    order_item_total: float = Field(163.99, alias="Order Item Total")
    order_profit_per_order: float = Field(31.52, alias="Order Profit Per Order")
    product_price: float = Field(59.99, alias="Product Price")
    order_month: int = Field(1, ge=1, le=12, description="Month of order (1–12)")
    order_dow: int = Field(0, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")

    model_config = {"populate_by_name": True}

    @field_validator("shipping_mode")
    @classmethod
    def validate_shipping_mode(cls, v: str) -> str:
        valid = {"Standard Class", "Second Class", "First Class", "Same Day"}
        if v not in valid:
            raise ValueError(f"shipping_mode must be one of {sorted(valid)}")
        return v

    @field_validator("market")
    @classmethod
    def validate_market(cls, v: str) -> str:
        valid = {"LATAM", "Europe", "Pacific Asia", "USCA", "Africa"}
        if v not in valid:
            raise ValueError(f"market must be one of {sorted(valid)}")
        return v

    @field_validator("customer_segment")
    @classmethod
    def validate_customer_segment(cls, v: str) -> str:
        valid = {"Consumer", "Corporate", "Home Office"}
        if v not in valid:
            raise ValueError(f"customer_segment must be one of {sorted(valid)}")
        return v


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Probability of late delivery (0–1)")
    risk_tier: str = Field(..., description="low | medium | high")


class BulkPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    num_features: list[str]
    cat_features: list[str]
    risk_tier_thresholds: dict[str, str]


# ── Helper: build DataFrame from inputs ──────────────────────────────────────

_NUM_COL_MAP = {
    "days_for_shipment_scheduled": "Days for shipment (scheduled)",
    "benefit_per_order":           "Benefit per order",
    "sales_per_customer":          "Sales per customer",
    "order_item_discount":         "Order Item Discount",
    "order_item_discount_rate":    "Order Item Discount Rate",
    "order_item_product_price":    "Order Item Product Price",
    "order_item_profit_ratio":     "Order Item Profit Ratio",
    "order_item_quantity":         "Order Item Quantity",
    "sales":                       "Sales",
    "order_item_total":            "Order Item Total",
    "order_profit_per_order":      "Order Profit Per Order",
    "product_price":               "Product Price",
    "order_month":                 "order_month",
    "order_dow":                   "order_dow",
}

_CAT_COL_MAP = {
    "shipping_mode":    "Shipping Mode",
    "market":           "Market",
    "order_region":     "Order Region",
    "customer_segment": "Customer Segment",
    "department_name":  "Department Name",
    "category_name":    "Category Name",
    "payment_type":     "Type",
}


def _to_dataframe(shipments: list[ShipmentInput]) -> pd.DataFrame:
    rows = []
    for s in shipments:
        row = {}
        for field, col in _NUM_COL_MAP.items():
            row[col] = getattr(s, field)
        for field, col in _CAT_COL_MAP.items():
            row[col] = getattr(s, field)
        rows.append(row)

    feature_order = (
        _model_payload["num_features"] + _model_payload["cat_features"]
    )
    return pd.DataFrame(rows)[feature_order]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check — confirms the API is running and the model is loaded."""
    loaded = bool(_model_payload)
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_name=_model_payload.get("model_name", "unknown"),
        model_loaded=loaded,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
def metrics() -> MetricsResponse:
    """Returns model performance stats recorded at training time."""
    if not _model_payload:
        raise HTTPException(status_code=503, detail="Model not loaded")
    m = _model_payload["metrics"]
    return MetricsResponse(
        model_name=_model_payload["model_name"],
        accuracy=round(m["Accuracy"], 4),
        precision=round(m["Precision"], 4),
        recall=round(m["Recall"], 4),
        f1=round(m["F1"], 4),
        auc_roc=round(m["AUC-ROC"], 4),
        num_features=_model_payload["num_features"],
        cat_features=_model_payload["cat_features"],
        risk_tier_thresholds={
            "low":    "score < 0.40",
            "medium": "0.40 ≤ score < 0.70",
            "high":   "score ≥ 0.70",
        },
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(shipment: ShipmentInput) -> PredictionResponse:
    """
    Predict late-delivery risk for a single shipment.

    Returns `risk_score` (0–1 probability) and `risk_tier` (low/medium/high).
    """
    if not _model_payload:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = _to_dataframe([shipment])
        prob = float(_pipeline().predict_proba(df)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictionResponse(risk_score=round(prob, 4), risk_tier=_tier(prob))


@app.post("/bulk_predict", response_model=BulkPredictionResponse, tags=["prediction"])
def bulk_predict(shipments: list[ShipmentInput]) -> BulkPredictionResponse:
    """
    Predict late-delivery risk for a batch of shipments.

    Accepts a JSON array of shipment objects (same schema as `/predict`).
    Returns predictions in the same order as the input list.
    """
    if not _model_payload:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not shipments:
        raise HTTPException(status_code=422, detail="shipments list must not be empty")
    if len(shipments) > 10_000:
        raise HTTPException(status_code=422, detail="Maximum 10,000 shipments per request")

    t0 = time.perf_counter()
    try:
        df = _to_dataframe(shipments)
        probs = _pipeline().predict_proba(df)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    predictions = [
        PredictionResponse(risk_score=round(float(p), 4), risk_tier=_tier(float(p)))
        for p in probs
    ]
    return BulkPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        elapsed_ms=round(elapsed_ms, 2),
    )
