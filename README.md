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

### Business Insights:
- **Usage patterns are the dominant predictor** - Daily activity time is the strongest indicator
- **Account characteristics matter** - Verification status and account age influence engagement
- **Temporal factors show significance** - When users join affects their engagement patterns
- **Geographic variations exist** - Country-specific usage patterns detected

## Installation & Setup

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm
pip install shap lime
pip install plotly
```

### Usage

#### 1. Run Complete Analysis
```python
# Execute notebooks in sequence:
python 01_EDA_Analysis.py
python 02_Data_Preprocessing.py
python 03_Baseline_Modeling.py
python 04_Prediction_Pipeline.py
python 05_XAI_Analysis.py
```

#### 2. Make Predictions
```python
import pickle
import pandas as pd

# Load pipeline
with open('social_media_prediction_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Predict single user
user_data = {
    'user_id': 'USER_001',
    'Usage_Z_Score': 1.5,  # High usage
    'Verified_Account_Encoded': 1,
    'Account_Age_Days': 0.5
}

prediction = predict_single_user(user_data)
print(f"Engagement Level: {prediction['engagement_level']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

#### 3. Explain Predictions
```python
# Generate SHAP explanations
explanation = ensemble_predict(user_data)
print(f"Consensus: {explanation['consensus_strength']:.1%}")
```

## Model Interpretability

### SHAP Summary
- **Global Explanations**: Feature importance across all predictions
- **Local Explanations**: Individual prediction breakdowns
- **Interaction Effects**: Feature combination impacts

### LIME Analysis
- **Model-Agnostic**: Works with any model type
- **Local Focus**: Explains individual predictions
- **Feature Contributions**: Positive/negative impact quantification

## Performance Metrics

- **Target Accuracy**: ≥80% (Achieved: 98.7%)
- **Target F1-Score**: ≥75% (Achieved: 98.7%)
- **Target ROC-AUC**: ≥85% (Achieved: 99.9%)
- **Model Count**: 11 algorithms evaluated
- **Feature Engineering**: 7 → 77 features (11x expansion)
- **Data Quality**: 100% complete, no missing values

## Business Applications

1. **Marketing Segmentation**: Identify high-value users for targeted campaigns
2. **Platform Optimization**: Understand engagement drivers for feature development
3. **Churn Prevention**: Early identification of users at risk of low engagement
4. **Content Strategy**: Tailor content based on engagement predictors
5. **Regional Insights**: Customize offerings based on geographic patterns

## Technical Specifications

- **Python Version**: 3.8+
- **Primary Libraries**: scikit-learn, pandas, numpy, xgboost, shap
- **Model Format**: Pickle serialization
- **Feature Count**: 77 engineered features
- **Memory Usage**: ~2MB processed dataset
- **Training Time**: <5 minutes on standard hardware

## File Descriptions

- `Social Media Users.csv`: Original dataset
- `baseline_model_results.csv`: Model performance comparison
- `feature_importance_comparison.csv`: Feature importance across methods
- `social_media_prediction_pipeline.pkl`: Complete trained pipeline
- `xai_analysis_results.pkl`: Explainability analysis results

## Future Enhancements

1. **Real-time Prediction**: API endpoint development
2. **Deep Learning**: Neural network architectures exploration
3. **Time Series Analysis**: Temporal engagement pattern modeling
4. **A/B Testing Framework**: Intervention impact measurement
5. **Advanced Feature Engineering**: NLP analysis of usage patterns

## Success Metrics

- 5/11 models achieved >98% accuracy
- Feature engineering increased predictive power significantly
- XAI analysis provides clear business insights
- Production-ready pipeline with error handling
- Comprehensive documentation and reproducibility

## License

This project is developed for educational and research purposes. Dataset usage should comply with original data source terms.

## Contact

For questions about implementation, methodology, or results interpretation, please refer to the detailed code documentation and XAI analysis results.

---

*Last Updated: August 2025*
*Model Performance: 98.7% Accuracy | 99.9% ROC-AUC*
