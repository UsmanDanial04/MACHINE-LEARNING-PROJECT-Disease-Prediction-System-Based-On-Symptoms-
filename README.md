# 🫀 Disease Prediction System Based On Symptoms

A supervised machine learning project that predicts the likelihood of **cardiovascular disease** in a patient based on clinical and lifestyle features such as blood pressure, cholesterol, BMI, and more.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Overview

This project builds a binary classification model to predict whether a person has cardiovascular disease (`1`) or not (`0`). The pipeline includes data cleaning, feature engineering, exploratory data analysis, model training, and single-patient prediction.

The model is trained on a dataset of **70,000 patient records** using **Logistic Regression**, achieving meaningful accuracy on held-out test data.

---

## Dataset

| Property | Details |
|---|---|
| **File** | `Dataset.csv` |
| **Records** | 70,000 patients |
| **Features** | 13 columns (12 input + 1 target) |
| **Separator** | Semicolon (`;`) |
| **Target** | `cardio` — `0` = No disease, `1` = Has disease |
| **Class Balance** | ~50/50 (35,021 vs 34,979) |

### Feature Dictionary

| Column | Description | Encoding |
|---|---|---|
| `id` | Unique row identifier | — |
| `age` | Age in days → converted to years | Continuous |
| `gender` | Biological sex | `1` = Female, `2` = Male |
| `height` | Height in centimetres | Continuous |
| `weight` | Weight in kilograms | Continuous |
| `ap_hi` | Systolic blood pressure | Continuous |
| `ap_lo` | Diastolic blood pressure | Continuous |
| `cholesterol` | Cholesterol level | `1` = Normal, `2` = Above normal, `3` = Well above normal |
| `gluc` | Glucose level | `1` = Normal, `2` = Above normal, `3` = Well above normal |
| `smoke` | Smoking status | `0` = Non-smoker, `1` = Smoker |
| `alco` | Alcohol consumption | `0` = No, `1` = Yes |
| `active` | Physical activity | `0` = Not active, `1` = Active |
| `cardio` | **Target** — cardiovascular disease | `0` = No, `1` = Yes |

---

## Project Structure

```
Disease-Prediction-System/
│
├── Dataset.csv          # Raw patient data (semicolon-separated)
├── ML_Project.ipynb     # Main Jupyter notebook (full pipeline)
└── README.md            # Project documentation
```

---

## Features

- **Data Cleaning** — removes outliers in blood pressure, height, and weight; drops duplicates
- **Feature Engineering** — converts age from days to years; derives BMI; categorises BMI into 6 clinical categories
- **Exploratory Data Analysis (EDA)** — countplots, boxplots, histograms, pie charts, and a correlation heatmap
- **Model Training** — Logistic Regression with StandardScaler normalisation
- **Evaluation** — accuracy score, confusion matrix, and classification report
- **Single-Patient Prediction** — accepts a new patient's data and outputs a prediction in real time

---

## Methodology

### 1. Data Cleaning
- Renamed columns for clarity (`ap_hi` → `Systolic_Pressure`, etc.)
- Filtered physiologically impossible values:
  - Systolic pressure: 80–250 mmHg
  - Diastolic pressure: 40–200 mmHg
  - Height: 120–220 cm
  - Weight: 40–150 kg
- Confirmed zero null values and zero duplicate rows

### 2. Feature Engineering
```python
# Age conversion
cardio['age_in_years'] = (cardio['age'] / 365).astype(int)

# BMI calculation
cardio['BMI'] = cardio['weight'] / ((cardio['height'] / 100) ** 2)

# BMI categorisation
bmi_bins  = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
bmi_labels = [0, 1, 2, 3, 4, 5]   # Underweight → Obesity III
```

### 3. Train / Test Split & Scaling
- **80 / 20** train-test split (`random_state=42`)
- `StandardScaler` fitted on training data only; applied to both splits

### 4. Model
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
```

---

## Results

The model is evaluated on the 20% held-out test set using:

| Metric | Description |
|---|---|
| **Accuracy** | Overall percentage of correct predictions |
| **Confusion Matrix** | True/false positives and negatives |
| **Classification Report** | Precision, Recall, F1-score per class |

> Run the notebook to view the exact scores, as they depend on the final cleaned dataset size.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# 1. Clone or download the repository
git clone https://github.com/your-username/disease-prediction-system.git
cd disease-prediction-system

# 2. Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook ML_Project.ipynb
```

---

## Usage

### Running the Full Pipeline

Open `ML_Project.ipynb` and run all cells sequentially. The notebook will:

1. Load and explore `Dataset.csv`
2. Clean and engineer features
3. Visualise the data
4. Train and evaluate the Logistic Regression model
5. Predict on two sample patients

### Predicting for a New Patient

At the end of the notebook, replace the values in the `person` dictionary with your own patient data:

```python
person = {
    'id'                  : 999999,
    'gender'              : 1,      # 1 = Female, 2 = Male
    'height'              : 165,    # cm
    'weight'              : 70,     # kg
    'Systolic_Pressure'   : 120,    # mmHg
    'Diastolic_Pressure'  : 80,     # mmHg
    'cholesterol'         : 1,      # 1/2/3
    'Glucose'             : 1,      # 1/2/3
    'Smoker'              : 0,      # 0 or 1
    'Alcoholic'           : 0,      # 0 or 1
    'active'              : 1,      # 0 or 1
    'age_in_years'        : 45,
    'BMI'                 : 70 / ((165 / 100) ** 2),
    'BMI_category'        : 2       # 0–5
}
```

The model will output `0` (no disease) or `1` (disease detected).

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualisation |
| `seaborn` | Statistical visualisation |
| `scikit-learn` | ML model, scaling, and evaluation |

---

## Author

**Usman Danial**  
Student ID: F2023266622

---

> ⚠️ **Disclaimer:** This project is for educational purposes only. Predictions made by this model should **not** be used as a substitute for professional medical diagnosis.
