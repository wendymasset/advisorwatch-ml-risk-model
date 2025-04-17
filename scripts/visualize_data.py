# scripts/visualize_data.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from preprocess import load_data, preprocess_data

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

def get_full_dataframe():
    df_raw = load_data("data/advisorwatch_dataset.csv")
    X, y = preprocess_data(df_raw)
    df = X.copy()
    df["high_risk"] = y
    return df

def plot_high_risk_distribution(df):
    ax = sns.countplot(data=df, x='high_risk', hue='high_risk', palette='pastel', legend=False)
    plt.title("Distribution of High-Risk Advisors")
    plt.xlabel("High Risk (1 = Yes, 0 = No)")
    plt.ylabel("Count")

    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        percent = 100 * count / total
        ax.annotate(f'{percent:.1f}%', (p.get_x() + 0.3, p.get_height() + 10))

    plt.tight_layout()
    plt.savefig("outputs/high_risk_distribution.png")
    plt.clf()

def plot_complaints_per_year(df):
    ax = sns.boxplot(x='high_risk', y='complaints_per_year', data=df, palette='muted')
    plt.title("Complaints per Year by Risk Level")
    plt.xlabel("High Risk")
    plt.ylabel("Complaints per Year")

    medians = df.groupby('high_risk')['complaints_per_year'].median()
    for i, median in enumerate(medians):
        ax.text(i, median + 0.02, f'Median: {median:.2f}', ha='center', color='black')

    plt.tight_layout()
    plt.savefig("outputs/complaints_per_year_boxplot.png")
    plt.clf()

def plot_fines_distribution(df):
    ax = sns.histplot(data=df, x='fines_k', hue='high_risk', bins=40, log_scale=(False, True), kde=True)
    plt.title("Fines Distribution by Risk Category")
    plt.xlabel("Fines (in $1,000s)")
    plt.ylabel("Frequency (Log Scale)")

    plt.annotate("High-risk advisors skew right\nwith heavier fines", xy=(80, 10), xytext=(50, 30),
                 arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10)

    plt.tight_layout()
    plt.savefig("outputs/fines_distribution_log.png")
    plt.clf()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.clf()

# Optional: run everything if script is executed directly
if __name__ == "__main__":
    df = get_full_dataframe()
    plot_high_risk_distribution(df)
    plot_complaints_per_year(df)
    plot_fines_distribution(df)
    plot_correlation_heatmap(df)