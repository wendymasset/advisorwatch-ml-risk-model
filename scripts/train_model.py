# train_model.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from preprocess import load_data, preprocess_data


def main():
    os.makedirs("outputs", exist_ok=True)
    # Load and preprocess
    df = load_data('data/advisorwatch_dataset.csv')
    X, y = preprocess_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train models
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predict
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # Evaluate
    print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))
    print("Random Forest:\n", classification_report(y_test, y_pred_rf))

    # Feature Importance
    importances = rf.feature_importances_
    features = X.columns
    pd.Series(importances, index=features).nlargest(10).plot(kind='barh')
    plt.title("Feature Importance - Random Forest")
    plt.savefig("outputs/feature_importance.png")
    plt.show()

    # Save models
    joblib.dump(lr, 'outputs/logistic_regression_model.pkl')
    joblib.dump(rf, 'outputs/random_forest_model.pkl')

if __name__ == "__main__":
    main()