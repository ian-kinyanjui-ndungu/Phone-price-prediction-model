import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(df):
    """Clean the Xiaomi phone dataset."""
    # Create a copy of the dataframe
    df = df.copy()
    
    # Clean price (remove ₹ and convert to float)
    df['price'] = df['price'].str.replace('₹', '').str.replace(',', '').astype(float)
    
    # Extract RAM and Storage from storage_ram column
    df['ram_gb'] = df['storage_ram'].str.extract(r'RAM(\d+)').astype(float)
    df['storage_gb'] = df['storage_ram'].str.extract(r'Internal Storage(\d+)').astype(float)
    
    # Extract processor speed from os_processor
    df['processor_speed'] = df['os_processor'].str.extract(r'Primary Clock Speed([\d.]+)').astype(float)
    
    # Extract Android version
    df['android_version'] = df['os_processor'].str.extract(r'Android (\d+)')
    df['android_version'] = df['android_version'].fillna('10')  # Default to 10 if not found
    df['android_version'] = df['android_version'].astype(float)
    
    # Extract battery capacity
    df['battery_capacity'] = df['battery'].str.extract(r'(\d+)').astype(float)
    
    # Create 5G feature
    df['has_5g'] = df['network'].str.contains('5G').astype(int)
    
    # Create ratings feature
    df['ratings'] = df['ratings'].astype(float)
    
    # Select relevant features
    feature_columns = ['ram_gb', 'storage_gb', 'processor_speed', 'android_version', 
                      'battery_capacity', 'has_5g', 'ratings']
    
    return df[feature_columns + ['price']]

def prepare_features(df):
    """Prepare features for model training."""
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler