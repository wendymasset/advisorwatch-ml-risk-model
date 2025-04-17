# preprocess.py

import pandas as pd
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    if 'advisor_id' in df.columns:
        df = df.drop(columns=['advisor_id'])

    # Normalize fines
    if 'fines_total_usd' in df.columns:
        df['fines_k'] = df['fines_total_usd'] / 1000
        df = df.drop(columns=['fines_total_usd'])

    # Fill missing values
    df = df.fillna(0)

    # Complaints per year
    df['complaints_per_year'] = df['complaints_count'] / df['years_active']
    df['complaints_per_year'] = df['complaints_per_year'].replace([np.inf, -np.inf], 0)

    # High fines flag
    df['has_high_fines'] = (df['fines_k'] > 50).astype(int)

    # Many employers flag
    df['has_many_employers'] = (df['employment_changes'] > 5).astype(int)

    # Many disclosures flag
    df['has_multiple_disclosures'] = (df['disclosures_count'] > 3).astype(int)

    # Target + features
    X = df.drop('high_risk', axis=1)
    y = df['high_risk']

    return X, y