import pandas as pd

def load_xiaomi_data():
    """Load the Xiaomi phone data from local CSV file."""
    try:
        df = pd.read_csv('data/mi_data.csv')
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def verify_data():
    """Verify that the Xiaomi data was loaded correctly."""
    try:
        df = pd.read_csv('data/mi_data.csv')
        print("\nData verification:")
        print(f"Number of records: {len(df)}")
        print("\nFirst few records:")
        print(df.head())
        print("\nData columns:")
        print(df.columns.tolist())
        print("\nMissing values:")
        print(df.isnull().sum())
        return True
    except Exception as e:
        print(f"Error verifying data: {str(e)}")
        return False