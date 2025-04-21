import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with loaded data
    """
    try:
        # First try to load with comma separator which is standard
        df = pd.read_csv(file_path)
        
        # Check if all data was correctly parsed (if only one column was created, it might be using a different separator)
        if len(df.columns) <= 1:
            # Try with semicolon separator
            df = pd.read_csv(file_path, sep=';')
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def convert_scientific_notation(df):
    """
    Convert scientific notation strings to float values
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with converted values
    """
    df_converted = df.copy()
    
    # Process all columns
    for col in df_converted.columns:
        # Check if the column is of string type
        if df_converted[col].dtype == 'object':
            # Replace commas with points in scientific notation (e.g., 1,23E+08 -> 1.23E+08)
            if df_converted[col].str.contains('E', case=True, na=False).any() or \
               df_converted[col].str.contains('e', case=True, na=False).any():
                df_converted[col] = df_converted[col].astype(str).str.replace(',', '.', regex=False)
        
        # Try to convert to numeric
        try:
            df_converted[col] = pd.to_numeric(df_converted[col])
        except:
            # If conversion fails, keep as is
            pass
    
    return df_converted

def detect_and_handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Detect and handle outliers in the dataset
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check for outliers (if None, all numeric columns are used)
        method: 'iqr' for Interquartile Range method
        threshold: threshold for outlier detection (default: 1.5 for IQR)
        
    Returns:
        DataFrame with handled outliers
    """
    if columns is None:
        # Get only numeric columns
        columns = df.select_dtypes(include=np.number).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap the outliers (capping method)
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    return df_clean

def normalize_features(df, columns=None, scaler=None):
    """
    Normalize features using Min-Max scaling
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize (if None, all numeric columns except target are used)
        scaler: a pre-fitted scaler (if None, a new one is created)
        
    Returns:
        DataFrame with normalized features and the scaler object
    """
    if columns is None:
        # Get only numeric columns
        columns = df.select_dtypes(include=np.number).columns
    
    df_normalized = df.copy()
    
    # If no scaler is provided, create a new one
    if scaler is None:
        scaler = MinMaxScaler()
        df_normalized[columns] = scaler.fit_transform(df[columns])
    else:
        # Use the provided scaler for transformation
        df_normalized[columns] = scaler.transform(df[columns])
    
    return df_normalized, scaler

def prepare_data(file_path, target_column='Ethanol concentration', test_size=0.2, random_state=42):
    """
    Prepare data for machine learning models
    
    Args:
        file_path: Path to the CSV file
        target_column: Name of the target column
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Load data
    df = load_data(file_path)
    
    if df is None:
        return None, None, None, None, None
    
    # Convert scientific notation
    df = convert_scientific_notation(df)
    
    # Handle outliers
    df = detect_and_handle_outliers(df)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Normalize features
    feature_columns = X_train.columns
    X_train_normalized, scaler = normalize_features(X_train, columns=feature_columns)
    X_test_normalized, _ = normalize_features(X_test, columns=feature_columns, scaler=scaler)
    
    return X_train_normalized, X_test_normalized, y_train, y_test, scaler, feature_columns

def preprocess_input_data(input_data, scaler, feature_columns):
    """
    Preprocess input data for prediction
    
    Args:
        input_data: Dictionary with input features
        scaler: Fitted MinMaxScaler
        feature_columns: List of feature column names
        
    Returns:
        Preprocessed input data as numpy array
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0.0
    
    # Select only the required features and in the correct order
    input_df = input_df[feature_columns]
    
    # Normalize using the fitted scaler
    input_normalized = scaler.transform(input_df)
    
    return input_normalized
