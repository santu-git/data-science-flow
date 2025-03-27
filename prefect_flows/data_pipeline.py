import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prefect import flow, task
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import boto3
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the variables
access_key = os.getenv("DO_SPACES_ACCESS_KEY")
secret_key = os.getenv("DO_SPACES_SECRET_KEY")

# Define paths
RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_data.csv")
EDA_OUTPUT_PATH = Path("notebooks/eda_plots/")
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
RAW_DATA_DIR = Path("data/raw/")

# DigitalOcean Spaces configuration
DO_SPACES_BUCKET = "cred-card-data"
DO_SPACES_REGION = "blr1"
DO_SPACES_RAW_FOLDER = "raw-data/"
DO_SPACES_PROCESSED_FOLDER = "processed-data/"
DO_SPACES_EDA_FOLDER = "eda-outputs/"

# Task: Upload file to DigitalOcean Spaces
@task
def upload_to_spaces(file_path, folder):
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        region_name=DO_SPACES_REGION,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=f"https://{DO_SPACES_REGION}.digitaloceanspaces.com",  # Explicitly set DigitalOcean Spaces endpoint
    )
    s3.upload_file(
        str(file_path),
        DO_SPACES_BUCKET,
        f"{folder}{file_path.name}",
    )

# Task: Download dataset from Kaggle
@task
def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DATA_DIR), unzip=True)
    upload_to_spaces(RAW_DATA_PATH, DO_SPACES_RAW_FOLDER)
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
    upload_to_spaces(PROCESSED_DATA_PATH, DO_SPACES_PROCESSED_FOLDER)
    return df

# Task: Perform EDA
@task
def perform_eda(df):
    EDA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Histogram
    df.hist(figsize=(20, 15))
    hist_path = EDA_OUTPUT_PATH / "histograms.png"
    plt.savefig(hist_path)
    upload_to_spaces(hist_path, DO_SPACES_EDA_FOLDER)
    plt.close()

    # Boxplot
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=df.iloc[:, :-1])
    boxplot_path = EDA_OUTPUT_PATH / "boxplots.png"
    plt.savefig(boxplot_path)
    upload_to_spaces(boxplot_path, DO_SPACES_EDA_FOLDER)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    heatmap_path = EDA_OUTPUT_PATH / "correlation_heatmap.png"
    plt.savefig(heatmap_path)
    upload_to_spaces(heatmap_path, DO_SPACES_EDA_FOLDER)
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
