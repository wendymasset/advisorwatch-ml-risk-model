# AdvisorWatch: Predicting High-Risk Financial Advisors with Machine Learning

AdvisorWatch is a machine learning project that assesses the risk level of financial advisors based on complaint history, fines, disclosures, and career activity. It simulates a real-world compliance tool that could support fraud detection or early intervention within the financial industry.

This project demonstrates end-to-end data handling — from cleaning and feature engineering to model building, evaluation, and visual storytelling.

---

## Dataset

A synthetic dataset was generated to simulate real-world advisor profiles. Each record includes:

- `years_active`: years in the industry  
- `complaints_count`: total complaints filed  
- `fines_total_usd`: total fines issued  
- `disclosures_count`: regulatory disclosures  
- `employment_changes`: number of firms worked at  
- `regulatory_events`, `criminal_events`, `customer_disputes`: binary flags  
- `high_risk`: target variable (1 = high risk, 0 = low risk)

---

## Features & Techniques

### Data Preprocessing
- Removed unnecessary columns and normalized fine amounts
- Engineered features such as:
  - `complaints_per_year`
  - `has_high_fines`
  - `has_many_employers`
  - `has_multiple_disclosures`

### Model Training
- Trained both **Logistic Regression** and **Random Forest** models
- Evaluated using precision, recall, and F1-score
- Visualized feature importance for interpretability

### Visual Storytelling
Created Seaborn plots to highlight:
- Class distribution (`high_risk`)
- Complaints per year across risk levels
- Fines distribution (log-scaled)
- Correlation heatmap


### Tech Stack
- Python 3.10+
- Pandas – data manipulation and analysis
- NumPy – numerical operations
- scikit-learn – ML models and evaluation
- Seaborn – statistical visualization
- Matplotlib – plotting engine
- Joblib – model saving

### Future Improvements
- Add hyperparameter tuning and cross-validation
- Include real-world data (if accessible)
- Evaluate with ROC curves, AUC, and confusion matrix visuals
- Add unit tests for preprocessing and evaluation functions
- Add dashboard with data visualization tool for interactive insights


