
┌──────────────────────────┐
│        Raw Data          │
│ ──────────────────────   │
│ train.csv                │
│ test.csv                 │
│ sample_submission.csv    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Data Concatenation    │
│ ──────────────────────   │
│ train + test combined    │
│ (is_train flag added)    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Feature Engineering &    │
│ Preprocessing            │
│ ──────────────────────   │
│ Ordinal Encoding         │
│  • sleep_quality         │
│  • facility_rating       │
│  • exam_difficulty       │
│                          │
│ Binary Encoding          │
│  • internet_access       │
│                          │
│ Interaction Features     │
│  • study_efficiency      │
│  • rest_factor           │
│  • study_sleep_ratio     │
│                          │
│ Label Encoding           │
│  • gender                │
│  • course                │
│  • study_method          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Train / Test Split     │
│ ──────────────────────   │
│ X (features)             │
│ y (exam_score)           │
│ X_test                   │
└────────────┬─────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│          K-Fold Cross Validation (10 Folds)         │
│ ──────────────────────────────────────────────────  │
│                                                     │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │
│  │  XGBoost   │  │ LightGBM   │  │  CatBoost    │   │
│  │ Regressor  │  │ Regressor  │  │ Regressor    │   │
│  └─────┬──────┘  └─────┬──────┘  └─────┬────────┘   │
│        │               │               │            │
│  OOF Predictions   OOF Predictions  OOF Predictions │
│        │               │               │            │
│        └───────┬───────┴───────┬───────┘            │
│                ▼               ▼                    │
│           Test Predictions (Averaged)               │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌──────────────────────────┐
│     Stacking (Level-2)   │
│ ──────────────────────   │
│ Input Features:          │
│  • OOF_XGB               │
│  • OOF_LGB               │
│  • OOF_CAT               │
│                          │
│ Meta-Model:              │
│  • Ridge Regression      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Final Predictions      │
│ ──────────────────────   │
│ • Weighted ensemble      │
│ • Prediction clipping    │
│   (0 – 100)              │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     Submission File      │
│ ──────────────────────   │
│      submission.csv      │ 
└──────────────────────────┘

