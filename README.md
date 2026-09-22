# Parkinson's Disease Prediction Using Data Analytics and Machine Learning

**Author: Avhishikta Maiti**

A supervised machine learning project that predicts Parkinson's disease from biomedical voice measurements using data analytics and ML classifiers — SVM, Random Forest, and Logistic Regression.

---

## Dataset

- **Name:** Parkinson's Disease Data Set
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **File:** `parkinsons.csv`
- **Records:** 195 voice recordings from 31 subjects (23 with Parkinson's, 8 healthy)
- **Features:** 22 biomedical voice measurements + 1 target column (`status`)
- **Target:** `status` — `1` = Parkinson's Positive, `0` = Healthy

---

## Project Description

This project builds a binary classification system to diagnose Parkinson's disease using voice signal features. The pipeline covers:

1. **Exploratory Data Analysis (EDA)** — class distribution, frequency feature histograms, jitter/shimmer box plots, HNR vs PPE scatter, correlation heatmap, and feature importance chart.
2. **Feature Selection** — Pearson correlation with the target (`|r| > 0.35`) selects 10 features from the original 22.
3. **Data Standardization** — `StandardScaler` embedded inside each `Pipeline` to prevent data leakage during cross-validation.
4. **Model Comparison** — 5-fold Stratified Cross-Validation with default hyperparameters.
5. **Hyperparameter Tuning** — `GridSearchCV` with `Pipeline`-based tuning for SVM, Random Forest, and Logistic Regression.
6. **Final Model** — Random Forest (`n_estimators=20`) saved with scaler and feature list for deployment.

### Selected Features (10)
`MDVP:Fo(Hz)`, `MDVP:Flo(Hz)`, `MDVP:Jitter(Abs)`, `MDVP:Shimmer`, `MDVP:APQ`, `HNR`, `spread1`, `spread2`, `D2`, `PPE`

---

## Technologies Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.x | Core language |
| NumPy | 2.3.5 | Numerical operations |
| Pandas | 2.3.3 | Data manipulation |
| scikit-learn | 1.7.2 | ML models, pipelines, CV, GridSearchCV |
| Matplotlib | 3.10.6 | Visualisations |
| Seaborn | 0.13.2 | Statistical plots |
| Jupyter Notebook | 1.1.1 | Interactive development |

---

## Results

### Default Cross-Validation (5-Fold Stratified)
| Model | Mean CV Accuracy |
|---|---|
| Random Forest | 87.16 % |
| SVM | 83.97 % |
| Logistic Regression | 81.39 % |

### After GridSearchCV Tuning
| Model | Best CV Accuracy | Best Parameters |
|---|---|---|
| SVM | 90.40 % | `C=10, kernel='rbf', gamma=0.1` |
| Random Forest | 88.43 % | `n_estimators=20` |
| Logistic Regression | 82.68 % | `C=10` |

### Test Set Performance
| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| SVM | 94.87 % | 93.55 % | 100.0 % | 96.67 % |
| **Random Forest** | **92.31 %** | **96.43 %** | **93.10 %** | **94.74 %** |
| Logistic Regression | 84.62 % | 87.10 % | 93.10 % | 90.00 % |

> **Final model chosen: Random Forest** — strong test accuracy (92.31 %), interpretable feature importances, and robust to feature scaling.

---

## Project Structure

```
├── AvhishiktaMaiti_Parkinson's Disease Prediction Using Data Analytics and Machine Learning.ipynb        # Main notebook — EDA, training, evaluation
├── parkinsons.csv           # Dataset
├── model.pkl                # Saved Random Forest model
├── scaler.pkl               # Saved StandardScaler
├── selected_features.pkl    # Saved list of 10 selected features
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Setup & Run Instructions

### 1. Clone / Download the repository
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook "Train Model.ipynb"
```

### 5. Run all cells
In Jupyter: **Kernel → Restart & Run All**

### 6. Load the saved model for inference
```python
import pickle, pandas as pd

model     = pickle.load(open('model.pkl', 'rb'))
scaler    = pickle.load(open('scaler.pkl', 'rb'))
features  = pickle.load(open('selected_features.pkl', 'rb'))

# Example input (10 values in feature order)
sample = [180.978, 155.495, 2e-05, 0.03852, 0.02877, 16.176, -5.6579, 0.3159, 3.0983, 0.2004]
df     = pd.DataFrame([sample], columns=features)   # named columns suppress UserWarning
scaled = scaler.transform(df)
prediction = model.predict(scaled)[0]
print("Parkinson's Positive" if prediction == 1 else "Healthy")
```

---

## Key Notes

- The `StandardScaler` is fitted **only on training data** and applied to test data — no data leakage.
- Feature selection is performed using **training-set correlation only** to avoid look-ahead bias.
- All cross-validation uses `StratifiedKFold(n_splits=5)` to preserve the 75/25 class imbalance across folds.

---