import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prefect import flow, task
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Define paths
RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_data.csv")
EDA_OUTPUT_PATH = Path("notebooks/eda_plots/")
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
RAW_DATA_DIR = Path("data/raw/")

# Task: Download dataset from Kaggle
@task
def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DATA_DIR), unzip=True)
    return RAW_DATA_PATH

# Task: Load dataset
@task
def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df

# Task: Preprocess data
@task
def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    # Normalize relevant features
    df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()
    # Save processed data
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df

# Task: Perform EDA
@task
def perform_eda(df):
    EDA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Histogram
    df.hist(figsize=(20, 15))
    plt.savefig(EDA_OUTPUT_PATH / "histograms.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=df.iloc[:, :-1])
    plt.savefig(EDA_OUTPUT_PATH / "boxplots.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.savefig(EDA_OUTPUT_PATH / "correlation_heatmap.png")
    plt.close()

# Prefect flow
@flow(log_prints=True)
def data_pipeline():
    raw_data_path = download_data()
    df = load_data()
    cleaned_df = preprocess_data(df)
    perform_eda(cleaned_df)

# Schedule the flow to run every 3 minutes
if __name__ == "__main__":
    data_pipeline()
