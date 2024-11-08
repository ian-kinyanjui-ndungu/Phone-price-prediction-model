import os
import pandas as pd
from src.data_processing import clean_data, prepare_features
from src.model import train_model, evaluate_model, make_prediction
from src.utils import load_xiaomi_data, verify_data
from sklearn.model_selection import train_test_split

def predict_phone_price(model, scaler):
    """Function to get user input and make predictions"""
    try:
        print("\nEnter phone specifications for prediction:")
        ram = float(input("RAM (GB): "))
        storage = float(input("Storage (GB): "))
        processor_speed = float(input("Processor Speed (GHz): "))
        android_version = float(input("Android Version (e.g., 13): "))
        battery = float(input("Battery Capacity (mAh): "))
        has_5g = int(input("Has 5G? (1 for Yes, 0 for No): "))
        rating = float(input("Expected Rating (1-5): "))

        # Create a DataFrame with the input data
        new_data = pd.DataFrame({
            'ram_gb': [ram],
            'storage_gb': [storage],
            'processor_speed': [processor_speed],
            'android_version': [android_version],
            'battery_capacity': [battery],
            'has_5g': [has_5g],
            'ratings': [rating]
        })

        # Make prediction
        predicted_price = make_prediction(model, scaler, new_data)
        print(f"\nPredicted phone price: ₹{predicted_price:.2f}")
    except ValueError as e:
        print(f"Invalid input: {str(e)}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

def main():
    try:
        # Verify the data file exists
        if not os.path.exists('data/mi_data.csv'):
            print("Error: mi_data.csv not found in data directory!")
            return

        # Load data
        print("Loading Xiaomi phone data...")
        df = load_xiaomi_data()
        if df is None:
            print("Failed to load data!")
            return

        # Verify the data
        if not verify_data():
            print("Data verification failed. Please check the data.")
            return

        # Clean data
        print("Cleaning data...")
        df_cleaned = clean_data(df)

        # Prepare features
        print("Preparing features...")
        X, y, scaler = prepare_features(df_cleaned)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        print("Training model...")
        model = train_model(X_train, y_train)

        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Interactive prediction loop
        while True:
            predict_phone_price(model, scaler)
            if input("\nMake another prediction? (y/n): ").lower() != 'y':
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()