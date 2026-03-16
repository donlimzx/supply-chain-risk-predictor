# Supply Chain Risk Predictor

This project was built end-to-end using Claude Code, with AI assistance across every phase — EDA, feature engineering, model training, API development, and dashboard design. This project predicts late-delivery risk for supply chain shipments using the DataCo Smart Supply Chain dataset (180k+ orders). Ships a trained XGBoost model behind a FastAPI REST API and a Streamlit analytics dashboard. 

---

## Business Problem

Late deliveries erode customer trust, inflate logistics costs, and distort inventory planning. This project answers: **given what is known at the moment an order is placed — shipping mode, destination market, contracted SLA days — how likely is this shipment to arrive late?**

The model surfaces a continuous risk score (0–1) and a human-readable tier (low / medium / high) so operations teams can intervene before a shipment departs rather than apologise after it arrives.

---

## Architecture

```
data/
└── DataCoSupplyChainDataset.csv   Raw source (180,519 rows, 53 columns)

notebooks/
└── 01_eda.ipynb                   Phase 1 — exploratory data analysis
    figures/
    ├── phase2_roc_cm.png          ROC curves + confusion matrices
    └── phase2_feature_importance.png

src/
├── train_model.py                 Phase 2 — feature engineering + model training
├── model.pkl                      Serialised XGBoost pipeline (joblib)
└── api.py                         Phase 3 — FastAPI prediction service

dashboard/
└── app.py                         Phase 4 — Streamlit analytics dashboard
```

**Data flow**

```
CSV ──► train_model.py ──► model.pkl
                                │
                                ▼
                            api.py  ◄──  POST /predict
                                         POST /bulk_predict

CSV ──────────────────────► dashboard/app.py  (direct read, no API)
```

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| Data | pandas, numpy | Ingestion, feature engineering |
| EDA | matplotlib, seaborn | Exploratory visualisation |
| Modelling | scikit-learn, XGBoost | Preprocessing pipeline, classifiers |
| Serialisation | joblib | Model persistence |
| API | FastAPI, Uvicorn, Pydantic | REST prediction service |
| Dashboard | Streamlit, Plotly | Interactive analytics |

---

## Key Findings

### Dataset
- **180,519 orders** across 5 markets (LATAM, Europe, Pacific Asia, USCA, Africa), Jan 2015 – Sep 2017
- **54.8% of shipments arrive late** — a mild class imbalance, handled with `scale_pos_weight` in XGBoost

### Leakage audit (critical)
Several columns encode the outcome and must never be used as features:

| Column | Reason |
|---|---|
| `Delivery Status` | 1:1 mapping to target |
| `Days for shipping (real)` | Known only post-delivery |
| `shipping date` | Known only post-delivery |
| `Order Status` | Set after fulfilment |

### Top predictors (safe, known at order time)

| Feature | Insight |
|---|---|
| **Shipping Mode** | Dominant signal — see table below |
| `Days for shipment (scheduled)` | SLA length correlates with late risk |
| **Market / Order Region** | Geographic variance in carrier reliability |
| Customer Segment, Category | Secondary categorical signals |

### Shipping Mode late rates

| Shipping Mode | Late Rate | Orders |
|---|---|---|
| First Class | **95.3%** | 27,814 |
| Second Class | 76.6% | 35,216 |
| Same Day | 45.7% | 9,737 |
| Standard Class | 38.1% | 107,752 |

> First Class is paradoxically the riskiest mode — likely because its aggressive SLA is routinely missed.

---

## Model Performance

Both models trained on an 80/20 stratified split. Only leakage-free features used.

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| Accuracy | 0.696 | **0.698** |
| Precision | 0.843 | **0.845** |
| Recall | 0.549 | **0.550** |
| F1 | 0.665 | **0.666** |
| **AUC-ROC** | 0.742 | **0.749** |

**XGBoost is saved as the production model.**

The moderate recall (~55%) reflects genuine uncertainty — without post-shipment signals the model can only act on order-time features. Precision is high (0.845): when the model flags a shipment as late, it is correct 84% of the time, making it actionable for triage workflows.

---

## Getting Started

### Prerequisites

- Python 3.11+
- macOS: `brew install libomp` (required by XGBoost)
- The raw dataset at `data/DataCoSupplyChainDataset.csv` (not included in repo — see [DataCo dataset](https://data.mendeley.com/datasets/8gx2fvg2k6/5))

### Install dependencies

```bash
pip install -r requirements.txt
```

### 1 — Train the model

```bash
python src/train_model.py
```

Outputs `src/model.pkl` and two figures under `notebooks/figures/`.

### 2 — Run the API

```bash
uvicorn src.api:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

**Minimal prediction request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Shipping Mode": "First Class",
    "Days for shipment (scheduled)": 2,
    "Market": "Europe"
  }'
```

```json
{ "risk_score": 0.87, "risk_tier": "high" }
```

**Risk tiers:**

| Tier | Score range |
|---|---|
| low | < 0.40 |
| medium | 0.40 – 0.70 |
| high | ≥ 0.70 |

**All endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Model performance stats |
| `POST` | `/predict` | Single shipment risk score |
| `POST` | `/bulk_predict` | Batch prediction (up to 10,000) |

### 3 — Run the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

Dashboard includes: KPI cards, late rate by shipping mode, late rate by market and region, and a filterable shipment records table.

---

## Project Phases

| Phase | Deliverable | Description |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Data loading, null analysis, target exploration, leakage audit |
| 2 | `src/train_model.py` | Feature engineering, LR baseline, XGBoost, evaluation figures |
| 3 | `src/api.py` | FastAPI service with input validation and bulk inference |
| 4 | `dashboard/app.py` | Streamlit dashboard with filters, charts, and data table |
