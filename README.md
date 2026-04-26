# Diabetic Patient Readmission Risk Prediction

An end-to-end Machine Learning project to predict 30-day hospital readmission risk for diabetic patients using real-world clinical electronic health record (EHR) data.

---

## 📌 Problem Statement

Hospital readmission within 30 days is a key healthcare quality metric and a major driver of avoidable healthcare costs. Identifying patients at high risk of readmission enables clinical teams to implement timely post-discharge interventions (e.g., medication reconciliation, close outpatient follow-ups, and targeted patient education).

The objective of this project is to build and evaluate supervised machine learning models that predict whether a diabetic patient will be readmitted to the hospital within 30 days of discharge (`<30`), focusing on optimizing **Recall** and the **$F_2$-Score** to minimize missed high-risk patients.

---

## 📊 Dataset Overview

The project uses the **Diabetes 130-US Hospitals (1999–2008)** dataset from the UCI Machine Learning Repository.

- **Total Records:** 101,766 hospital encounters
- **Initial Features:** 50 clinical and demographic attributes
- **Target Variable:** `readmitted`
  - Positive Class (`1`): Readmission in `< 30` days (~11.16% of encounters)
  - Negative Class (`0`): Readmission in `> 30` days or `NO` readmission (~88.84% of encounters)
- **Class Imbalance Ratio:** ~1 : 8

---

## 🛠️ Project Pipeline

```
Raw EHR Data
    │
    ├── 1. Data Cleaning & Preprocessing
    │      ├── Age range conversion to numeric midpoints
    │      ├── Zero-variance feature removal (examide, citoglipton)
    │      ├── Missing value handling ('?' -> NaN)
    │      └── Highly sparse attribute pruning
    │
    ├── 2. Clinical Feature Engineering
    │      ├── ICD-9 primary & secondary diagnosis grouping (Circulatory, Respiratory, Digestive, Diabetes, Other)
    │      ├── Admission type & source mapping
    │      ├── Total healthcare utilization (inpatient + outpatient + emergency)
    │      ├── Clinical procedure & medication intensity ratios
    │      └── Log transformation on skewed utilization features
    │
    ├── 3. Patient-Grouped Split (Zero Data Leakage)
    │      └── GroupShuffleSplit (80% Train / 20% Test) grouped strictly by `patient_nbr`
    │
    ├── 4. Scikit-Learn Preprocessing Pipeline
    │      ├── Numerical: SimpleImputer (median) + StandardScaler
    │      └── Categorical: SimpleImputer (most_frequent) + OneHotEncoder (handle_unknown='ignore')
    │
    ├── 5. Manifold & Dimensionality Reduction Analysis
    │      ├── Variance Inflation Factor (VIF) collinearity check
    │      └── 2D t-SNE & UMAP manifold visualizations
    │
    └── 6. Supervised Modeling & Evaluation
           ├── Logistic Regression (class-weighted balanced)
           ├── Random Forest Classifier
           ├── XGBoost Classifier (scale_pos_weight)
           ├── CatBoost Classifier (native categorical handling & cost weighting)
           └── Hyperparameter Tuning (RandomizedSearchCV on CatBoost optimizing F2-Score)
```

---

## 🧪 Model Performance & Benchmarks

Because false negatives in clinical readmission are significantly more costly than false positives, models are evaluated with an emphasis on **Recall**, **$F_2$-Score** ($\beta=2$ weights recall higher than precision), **PR-AUC**, and **ROC-AUC**:

| Model | Accuracy | Precision | Recall | $F_1$-Score | $F_2$-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 0.6709 | 0.1611 | 0.4951 | 0.2431 | 0.3500 | 0.6358 | 0.1830 |
| **Random Forest (Baseline)** | 0.8932 | 0.3333 | 0.0005 | 0.0009 | 0.0006 | 0.6176 | 0.1668 |
| **XGBoost (`scale_pos_weight`)** | 0.6753 | 0.1640 | 0.4984 | 0.2468 | 0.3540 | 0.6339 | 0.1815 |
| **CatBoost (Baseline)** | 0.6478 | 0.1551 | 0.5170 | 0.2386 | 0.3525 | 0.6375 | 0.1862 |
| **CatBoost (Tuned via RandomizedSearchCV)** | **0.6451** | **0.1562** | **0.5281** | **0.2411** | **0.3578** | **0.6366** | **0.1849** |

---

## 📁 Repository Structure

```
.
├── diabetic_data.csv                  # Dataset containing 101,766 patient encounters
├── diabetic_patient-readmission.ipynb # Main end-to-end Jupyter Notebook
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git exclusion rules
└── README.md                          # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ or Python 3.10+ installed
- Git installed

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Notebook
```bash
jupyter notebook diabetic_patient-readmission.ipynb
```

---

## 💡 Key Clinical Takeaways

1. **Prior Utilization is the Strongest Predictor:** Previous inpatient and emergency visits correlate strongly with 30-day readmission risk.
2. **Preventing Patient Data Leakage:** Splitting by hospital encounters without grouping by `patient_nbr` leads to overoptimistic performance due to identical patient histories leaking into the test set. Using `GroupShuffleSplit` maintains clinical evaluation integrity.
3. **Metric Alignment:** Default decision boundaries and raw accuracy can be deceptive on imbalanced medical datasets; optimizing for $F_2$-score ensures clinical teams capture the maximum number of true readmissions.
