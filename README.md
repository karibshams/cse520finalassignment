# Social Media User Engagement Prediction

A comprehensive machine learning project for predicting social media user engagement levels using behavioral data, account characteristics, and temporal features.

## Project Overview

This project implements a complete end-to-end machine learning pipeline to classify social media users into "High Engagement" and "Low Engagement" categories based on their platform usage patterns. The system includes data preprocessing, feature engineering, model training, prediction capabilities, and explainable AI analysis.

## Dataset Information

- **Source**: Social Media Users Dataset (CSV format)
- **Size**: 10,000 user records
- **Features**: 7 original features expanded to 77+ engineered features
- **Target**: Binary classification (High/Low Engagement)
- **Threshold**: Users with >75th percentile daily usage time (≈180+ minutes) classified as High Engagement

### Original Features
- Daily Time Spent (min)
- Platform
- Owner
- Primary Usage
- Country
- Verified Account
- Date Joined

## Project Structure

```
├── data/
│   ├── Social Media Users.csv                    # Original dataset
│   ├── processed_social_media_dataset.csv        # Preprocessed features
│   └── processed_social_media_dataset_with_target.csv
├── models/
│   ├── social_media_prediction_pipeline.pkl      # Trained models
│   ├── baseline_model_results.csv               # Model performance
│   └── xai_analysis_results.pkl                 # XAI analysis results
├── notebooks/
│   ├── 01_EDA_Analysis.py                       # Exploratory Data Analysis
│   ├── 02_Data_Preprocessing.py                 # Feature Engineering
│   ├── 03_Baseline_Modeling.py                  # Model Training & Evaluation
│   ├── 04_Prediction_Pipeline.py               # Production Pipeline
│   └── 05_XAI_Analysis.py                      # Explainable AI
└── README.md
```

## Implementation Pipeline

### 1. Exploratory Data Analysis (EDA)
- Comprehensive dataset analysis and visualization
- Data quality assessment (no missing values, no duplicates)
- Feature type identification and statistical analysis
- Correlation analysis and outlier detection

### 2. Data Preprocessing & Feature Engineering
- **Temporal Features**: Date parsing, account age calculation, seasonal indicators
- **Categorical Encoding**: One-hot encoding for platforms, owners, usage types
- **Geographic Features**: Country frequency encoding and regional grouping
- **Interaction Features**: Usage intensity, verification status combinations
- **Statistical Features**: Z-scores, percentile rankings
- **Feature Scaling**: StandardScaler for numerical features

### 3. Model Development & Evaluation
Implemented and evaluated 11 machine learning algorithms:
- Logistic Regression
- Naive Bayes
- Decision Trees
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- Neural Network (MLP)

### 4. Model Performance
| Model | Accuracy | F1-Score | ROC-AUC | Status |
|-------|----------|----------|---------|---------|
| Random Forest | 0.987 | 0.987 | 0.999 | ✅ PASS |
| XGBoost | 0.986 | 0.986 | 0.999 | ✅ PASS |
| LightGBM | 0.985 | 0.985 | 0.999 | ✅ PASS |
| Gradient Boosting | 0.984 | 0.984 | 0.999 | ✅ PASS |
| Decision Trees | 0.979 | 0.979 | 0.979 | ✅ PASS |

### 5. Prediction Pipeline
Production-ready prediction system with:
- Single user prediction
- Batch prediction capabilities
- Ensemble prediction methods
- Error handling and validation

### 6. Explainable AI (XAI)
- **SHAP Analysis**: Global and local feature importance
- **LIME Analysis**: Model-agnostic local explanations
- **Feature Impact Visualization**: Waterfall plots, summary plots
- **Business Insights**: Actionable recommendations

## Key Findings

### Most Important Features (SHAP Analysis):
1. **Usage_Z_Score** (7.79 importance) - Standardized usage time
2. **Verified_High_Usage** (0.24) - Interaction between verification and usage
3. **Month_Joined** (0.08) - Temporal joining patterns
4. **Account_Age_Days** (0.05) - Account maturity
5. **Day_of_Week_Joined** (0.04) - Signup timing patterns

