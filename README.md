# AI-Powered Aadhaar Operations Intelligence Dashboard

---

## Overview

The AI-Powered Aadhaar Operations Intelligence Dashboard is an integrated decision-support system built using anonymised public Aadhaar datasets. The platform transforms static operational data into predictive insights using machine learning and statistical modelling techniques.

It combines enrolment trend analysis, demographic update forecasting, lifecycle transition modelling, and dormancy risk assessment into a unified interactive dashboard. The objective is to enable data-driven operational planning and strengthen inclusive identity governance.

This system operates entirely on aggregated, anonymised public datasets and does not use any personally identifiable information.

---

## Problem Statement

Public identity systems generate large volumes of operational data. However, such data often remains descriptive rather than predictive. Administrative planning can benefit significantly from early trend detection, workload forecasting, and risk identification.

This project addresses the need to:

- Detect enrolment slowdowns early  
- Forecast demographic update demand  
- Predict biometric transition workload  
- Identify potential identity dormancy risk  
- Provide confidence-aware predictions for responsible interpretation  

---

## Core Analytical Modules

---

### 1. Enrolment Momentum Score (EMS)

**Model:** Random Forest Classifier  

The Enrolment Momentum Score module predicts whether enrolment activity is increasing, declining, stable, or uncertain. It uses lag-based trend features derived from historical enrolment data.

**Key Features**

- Lag-based feature engineering (lag_1, lag_2, lag_3)  
- Probability-based classification  
- Confidence-aware prediction banding  
- Decision guidance interpretation  

**Performance**

Approximately 85% classification accuracy using time-aware validation splits.

**Purpose**

To enable early detection of enrolment acceleration or slowdown and support proactive operational adjustments.

---

### 2. Demographic Update Pressure (DUP)

**Models:**  
K-Means Clustering  
Linear Regression (state-specific models)

This module identifies states with high demographic update workload and forecasts near-term update demand.

**Capabilities**

- Segmentation of states into Low, Moderate, and High pressure clusters  
- Historical trend analysis  
- Demand projection using regression models  
- Update burden assessment  

**Purpose**

To support infrastructure planning and workload distribution decisions.

---

### 3. Lifecycle Transition Predictor (LTP)

**Model:** Linear Regression  

The Lifecycle Transition Predictor estimates biometric update demand driven by predictable age transitions, particularly the transition into the 17+ age category.

**Inputs**

- Age 5–17 cohort size  
- Time index variable  

**Outputs**

- Forecasted biometric update counts  

**Purpose**

To anticipate workload resulting from demographic lifecycle transitions and enable proactive capacity planning.

---

### 4. Aadhaar Dormancy Risk (ADR)

**Model:** K-Means Clustering (Unsupervised)

The Dormancy Risk module evaluates engagement behaviour using engineered ratios.

**Features**

- DER (Demographic Engagement Ratio)  
- BER (Biometric Engagement Ratio)  
- Composite ADR scoring metric  

**Risk Categories**

- Low Risk  
- Moderate Risk  
- High Risk  

**Purpose**

To identify potential disengagement risk within the identity ecosystem and support inclusive governance strategies.

---

## Datasets Used

All datasets are anonymised public datasets made available by UIDAI.

### Aadhaar Enrolment Dataset

**Columns Used**

- date  
- state  
- age_0_5  
- age_5_17  
- age_18_greater  

Used for enrolment trend modelling and momentum classification.

---

### Demographic Update Dataset

**Columns Used**

- date  
- state  
- update counts  

Used for pressure clustering and demand forecasting.

---

### Biometric Update Dataset

**Columns Used**

- date  
- state  
- bio_age_5_17  
- bio_age_17_  

Used for lifecycle transition modelling.

---

## Data Processing and Feature Engineering

Key preprocessing steps include:

- Date format normalisation  
- Monthly aggregation  
- State-level grouping  
- Missing value handling  
- Cross-dataset alignment  
- Lag feature generation  
- Ratio engineering (DER, BER)  
- Time index construction for forecasting  

**Engineered Features**

- lag_1, lag_2, lag_3 enrolment variables  
- Engagement ratios  
- Lifecycle transition signals  
- Update load averages  
- Time variable for regression modelling  

---

## Machine Learning Techniques Applied

The project integrates multiple modelling paradigms:

- Classification (Random Forest)  
- Clustering (K-Means)  
- Regression-based forecasting (Linear Regression)  

This multi-model architecture enables both predictive classification and quantitative demand estimation within a unified framework.

---

## System Architecture

1. Raw CSV dataset ingestion  
2. Data cleaning and aggregation  
3. Feature engineering  
4. Model training  
5. Model persistence using .pkl files  
6. Streamlit-based dashboard integration  
7. Interactive prediction interface  
8. Real-time inference  

The system follows a modular design, allowing independent training and updating of each analytical module.

---

## Technology Stack

**Programming Language**

- Python  

**Libraries**

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- joblib  

**Frontend Framework**

- Streamlit  

**Deployment**

- Local deployment via Streamlit  
- Modular model loading using persisted .pkl files  

---

## Performance Summary

**Enrolment Momentum Score**

- Approximately 85% classification accuracy  
- Strong performance in detecting increasing enrolment trends  
- Conservative uncertainty handling  

**Demographic Update Pressure**

- Meaningful clustering segmentation  
- Clear state-level demand projections  

**Lifecycle Transition Predictor**

- Stable forecasting driven by demographic patterns  
- Reliable near-term workload estimation  

**Dormancy Risk**

- Distinct engagement clusters  
- Practical risk categorisation  

---

## Societal Impact

The system supports:

- Early identification of enrolment decline  
- Proactive update infrastructure planning  
- Forecasting of biometric transition workload  
- Detection of potential dormancy risk  
- Prevention of service bottlenecks  
- Strengthening inclusive digital identity access  

By transforming descriptive datasets into predictive intelligence, the platform enhances evidence-based administrative decision-making.

---

## Ethical Considerations

- All datasets used are publicly available and anonymised  
- No personally identifiable information is processed  
- Models operate only on aggregated statistical data  
- Outputs are confidence-aware to reduce risk of misinterpretation  

---

## How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/FFTOPPER/Aadhar-UIDAI-Portal.git
cd Aadhar-UIDAI-Portal
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## Requirements

Python Version: 3.9+

### Recommended Setup

Create a virtual environment:

python -m venv venv

Activate the environment:

Mac/Linux:
source venv/bin/activate

Windows:
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt


## Future Enhancements

- Integration of advanced time-series models (ARIMA, Prophet)  
- Model explainability using SHAP  
- API-based deployment  
- Automated data refresh pipelines  
- Cloud-based deployment architecture  

---
