# Download Student Performance Dataset
import os
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Create data directory if it doesn't exist
data_dir = os.path.join("data", "raw")
os.makedirs(data_dir, exist_ok=True)

print("Downloading Student Performance Dataset from UCI ML Repository...")

try:
    # Fetch dataset
    student_performance = fetch_ucirepo(id=320)

    # Get features and targets
    X = student_performance.data.features
    y = student_performance.data.targets

    # Combine features and targets
    df = pd.concat([X, y], axis=1)

    # Save to CSV
    output_path = os.path.join(data_dir, "student_performance.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset downloaded successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"💾 Saved to: {output_path}")
    print(f"📋 Columns: {list(df.columns)}")

    # Display basic info
    print("\nDataset Info:")
    print(df.info())

    print("\nFirst 5 rows:")
    print(df.head())

except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("Please check your internet connection and try again.")
