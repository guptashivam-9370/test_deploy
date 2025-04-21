import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import requests
import json
from io import BytesIO

from preprocessing import load_data, convert_scientific_notation, detect_and_handle_outliers, normalize_features, prepare_data
from models import ModelTrainer
from utils import (create_feature_importance_plot, plot_correlation_matrix, 
                  plot_actual_vs_predicted, plot_feature_distributions,
                  get_table_download_link, create_model_comparison_plot)

# Page configuration
st.set_page_config(
    page_title="Ethanol Concentration Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables
API_URL = os.environ.get('API_URL', 'http://localhost:8000')
DATA_PATH = 'attached_assets/Shivam_dataset.csv'

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Model Evaluation", "Prediction"])

# Initialize session state for storing data and models
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'eval_metrics' not in st.session_state:
    st.session_state.eval_metrics = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None

def load_dataset():
    """Load and prepare the dataset"""
    data = load_data(DATA_PATH)
    if data is not None:
        data = convert_scientific_notation(data)
        st.session_state.data = data
        return data
    else:
        st.error(f"Failed to load dataset from {DATA_PATH}")
        return None

def train_and_evaluate():
    """Train and evaluate models"""
    with st.spinner("Preparing data..."):
        X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_data(
            DATA_PATH, 
            target_column='Ethanol concentration',
            test_size=0.2, 
            random_state=42
        )
        
        if X_train is None:
            st.error("Failed to prepare data")
            return
        
        # Store in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.feature_columns = feature_columns
    
    with st.spinner("Training models..."):
        # Initialize model trainer
        model_trainer = ModelTrainer()
        
        # Train models
        metrics = model_trainer.train_models(X_train, y_train, cv=5)
        
        # Evaluate models
        eval_metrics = model_trainer.evaluate_models(X_test, y_test)
        
        # Get best model
        best_model_name, _ = model_trainer.get_best_model(metric='R2')
        
        # Store in session state
        st.session_state.models = model_trainer
        st.session_state.metrics = metrics
        st.session_state.eval_metrics = eval_metrics
        st.session_state.best_model_name = best_model_name
        
        # Save models
        model_trainer.save_models(directory='saved_models')
        
        # Save scaler and feature columns
        os.makedirs('saved_models', exist_ok=True)
        joblib.dump(scaler, 'saved_models/scaler.joblib')
        joblib.dump(feature_columns, 'saved_models/feature_columns.joblib')

def predict_with_api(input_data):
    """Make prediction using the API"""
    try:
        response = requests.post(
            f"{API_URL}/predict", 
            json=input_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")
        return None

def predict_locally(input_data):
    """Make prediction locally using the trained model"""
    if (st.session_state.models is None or 
        st.session_state.scaler is None or 
        st.session_state.feature_columns is None or
        st.session_state.best_model_name is None):
        st.error("Models not trained yet. Please train models first.")
        return None
    
    try:
        # Get best model
        _, best_model = st.session_state.models.get_best_model(metric='R2')
        
        # Preprocess input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for col in st.session_state.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0
        
        # Select only the required features and in the correct order
        input_df = input_df[st.session_state.feature_columns]
        
        # Normalize using the fitted scaler
        input_normalized = st.session_state.scaler.transform(input_df)
        
        # Make prediction
        prediction = best_model.predict(input_normalized)[0]
        
        return {
            'status': 'success',
            'prediction': float(prediction),
            'model_used': st.session_state.best_model_name
        }
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# Page: Home
if page == "Home":
    st.title("🧪 Ethanol Concentration Predictor")
    
    st.markdown("""
    ### Welcome to the Ethanol Concentration Prediction Application!
    
    This application helps predict the ethanol concentration in distillation columns using various process parameters.
    
    #### Features:
    - Data exploration and visualization
    - Training of multiple regression models
    - Model evaluation and comparison
    - Interactive prediction
    
    #### Models Implemented:
    - Linear Regression
    - Lasso Regression
    - Ridge Regression
    - Support Vector Regression (SVR)
    - CatBoost Regressor
    
    #### How to Use:
    1. Start with the **Data Exploration** page to understand the dataset
    2. Go to **Model Training** to train the regression models
    3. Use **Model Evaluation** to compare model performance
    4. Finally, make predictions using the **Prediction** page
    
    Use the navigation menu on the left to get started!
    """)
    
    # Load dataset if not already loaded
    if st.session_state.data is None:
        if st.button("Load Dataset"):
            with st.spinner("Loading dataset..."):
                load_dataset()
                if st.session_state.data is not None:
                    st.success(f"Dataset loaded successfully! Shape: {st.session_state.data.shape}")
    else:
        st.success(f"Dataset already loaded! Shape: {st.session_state.data.shape}")
        
        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.data.head())

# Page: Data Exploration
elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    
    # Load dataset if not already loaded
    if st.session_state.data is None:
        with st.spinner("Loading dataset..."):
            data = load_dataset()
    else:
        data = st.session_state.data
    
    if data is not None:
        st.subheader("Dataset Information")
        
        # Display basic information
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Rows", data.shape[0])
        with col2:
            st.metric("Number of Columns", data.shape[1])
        
        # Display dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(data.describe())
        
        # Download link for dataset
        st.markdown(get_table_download_link(data, "ethanol_dataset.csv", "Download Full Dataset"), unsafe_allow_html=True)
        
        # Data visualizations
        st.subheader("Data Visualizations")
        
        viz_type = st.selectbox("Select Visualization", 
                               ["Feature Distributions", "Correlation Matrix", "Target Distribution", "Feature vs Target"])
        
        if viz_type == "Feature Distributions":
            fig = plot_feature_distributions(data)
            st.pyplot(fig)
        
        elif viz_type == "Correlation Matrix":
            fig = plot_correlation_matrix(data)
            st.pyplot(fig)
        
        elif viz_type == "Target Distribution":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data["Ethanol concentration"], kde=True, ax=ax)
            ax.set_title("Distribution of Ethanol Concentration")
            ax.set_xlabel("Ethanol Concentration")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        
        elif viz_type == "Feature vs Target":
            # Let user select a feature to plot against target
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            numeric_cols.remove("Ethanol concentration")  # Remove target from feature list
            
            selected_feature = st.selectbox("Select Feature", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=data[selected_feature], y=data["Ethanol concentration"], ax=ax)
            ax.set_title(f"{selected_feature} vs Ethanol Concentration")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Ethanol Concentration")
            st.pyplot(fig)

# Page: Model Training
elif page == "Model Training":
    st.title("🧠 Model Training")
    
    if st.session_state.models is None:
        st.info("Models have not been trained yet. Click the button below to start training.")
        
        # Training options
        with st.expander("Training Options"):
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 100, 42, 1)
        
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a few minutes."):
                train_and_evaluate()
            
            if st.session_state.models is not None:
                st.success("Models trained successfully!")
                st.balloons()
    else:
        st.success("Models have been trained!")
        
        # Display model metrics
        st.subheader("Model Performance (Cross-Validation)")
        
        metrics_df = pd.DataFrame({
            'Model': [],
            'CV MSE (mean)': [],
            'CV MSE (std)': [],
            'CV R² (mean)': [],
            'CV R² (std)': []
        })
        
        for model_name, metrics in st.session_state.metrics.items():
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Model': [model_name],
                'CV MSE (mean)': [metrics['CV_MSE_mean']],
                'CV MSE (std)': [metrics['CV_MSE_std']],
                'CV R² (mean)': [metrics['CV_R2_mean']],
                'CV R² (std)': [metrics['CV_R2_std']]
            })], ignore_index=True)
        
        st.dataframe(metrics_df)
        
        # Retrain models
        if st.button("Retrain Models"):
            with st.spinner("Retraining models... This may take a few minutes."):
                train_and_evaluate()
            
            st.success("Models retrained successfully!")
            st.balloons()

# Page: Model Evaluation
elif page == "Model Evaluation":
    st.title("📈 Model Evaluation")
    
    if (st.session_state.models is None or 
        st.session_state.eval_metrics is None or 
        st.session_state.X_test is None or 
        st.session_state.y_test is None):
        
        st.warning("Models haven't been trained yet. Please go to the Model Training page first.")
        
        if st.button("Train Models Now"):
            with st.spinner("Training models... This may take a few minutes."):
                train_and_evaluate()
            
            if st.session_state.models is not None:
                st.success("Models trained successfully!")
    else:
        # Display best model
        st.subheader("Best Performing Model")
        st.info(f"The best model based on R² score is: **{st.session_state.best_model_name}**")
        
        # Display evaluation metrics
        st.subheader("Model Performance (Test Set)")
        
        eval_df = pd.DataFrame({
            'Model': [],
            'MSE': [],
            'R²': []
        })
        
        for model_name, metrics in st.session_state.eval_metrics.items():
            eval_df = pd.concat([eval_df, pd.DataFrame({
                'Model': [model_name],
                'MSE': [metrics['MSE']],
                'R²': [metrics['R2']]
            })], ignore_index=True)
        
        st.dataframe(eval_df)
        
        # Visualize model comparison
        st.subheader("Model Comparison")
        
        metric_to_plot = st.selectbox("Select Metric for Comparison", ["R²", "MSE"])
        
        fig = create_model_comparison_plot(
            {model: metrics for model, metrics in st.session_state.eval_metrics.items()},
            metric_name=metric_to_plot
        )
        st.pyplot(fig)
        
        # Select model for detailed evaluation
        st.subheader("Detailed Model Evaluation")
        
        selected_model = st.selectbox("Select Model", list(st.session_state.models.trained_models.keys()))
        
        if selected_model:
            model = st.session_state.models.trained_models[selected_model]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature importance
                st.subheader("Feature Importance")
                if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
                    fig = create_feature_importance_plot(
                        model, 
                        st.session_state.feature_columns, 
                        selected_model
                    )
                    st.pyplot(fig)
                else:
                    st.info("Feature importance not available for this model type")
            
            with col2:
                # Actual vs Predicted
                st.subheader("Actual vs Predicted")
                y_pred = st.session_state.eval_metrics[selected_model]['Predictions']
                fig = plot_actual_vs_predicted(
                    st.session_state.y_test, 
                    y_pred,
                    selected_model
                )
                st.pyplot(fig)
            
            # Residuals
            st.subheader("Residuals Analysis")
            
            y_pred = st.session_state.eval_metrics[selected_model]['Predictions']
            residuals = st.session_state.y_test - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Residuals distribution
            sns.histplot(residuals, kde=True, ax=ax1)
            ax1.set_title("Distribution of Residuals")
            ax1.set_xlabel("Residual Value")
            ax1.set_ylabel("Frequency")
            
            # Residuals vs Predicted
            sns.scatterplot(x=y_pred, y=residuals, ax=ax2)
            ax2.axhline(y=0, color='r', linestyle='-')
            ax2.set_title("Residuals vs Predicted Values")
            ax2.set_xlabel("Predicted Ethanol Concentration")
            ax2.set_ylabel("Residual")
            
            plt.tight_layout()
            st.pyplot(fig)

# Page: Prediction
elif page == "Prediction":
    st.title("🔮 Prediction")
    
    # Check if models are trained
    models_available = (
        st.session_state.models is not None or 
        os.path.exists('saved_models/catboost_regressor.joblib')
    )
    
    if not models_available:
        st.warning("Models haven't been trained yet. Please go to the Model Training page first.")
        
        if st.button("Train Models Now"):
            with st.spinner("Training models... This may take a few minutes."):
                train_and_evaluate()
            
            if st.session_state.models is not None:
                st.success("Models trained successfully!")
    else:
        st.subheader("Predict Ethanol Concentration")
        
        # Load dataset if not already loaded to get feature names
        if st.session_state.data is None:
            with st.spinner("Loading dataset..."):
                data = load_dataset()
        else:
            data = st.session_state.data
        
        if data is not None:
            # Get feature names (excluding target)
            feature_names = data.drop(columns=['Ethanol concentration']).columns.tolist()
            
            # Input method selection
            input_method = st.radio("Input Method", ["Manual Entry", "Sample from Dataset"])
            
            input_data = {}
            
            if input_method == "Manual Entry":
                # Create a form for input
                with st.form("prediction_form"):
                    # Divide features into columns
                    col1, col2, col3 = st.columns(3)
                    
                    # Temperature columns (T1-T14)
                    temp_cols = [col for col in feature_names if col.startswith('T')]
                    
                    # Flow rate columns
                    flow_cols = [col for col in feature_names if col in ['L', 'V', 'D', 'B', 'F']]
                    
                    # Pressure column
                    pressure_col = ['Pressure']
                    
                    # Create input fields
                    with col1:
                        st.subheader("Pressure")
                        for col in pressure_col:
                            input_data[col] = st.number_input(f"{col}", value=data[col].mean())
                        
                        st.subheader("Flow Rates")
                        for col in flow_cols:
                            input_data[col] = st.number_input(f"{col}", value=data[col].mean())
                    
                    with col2:
                        st.subheader("Temperatures (T1-T7)")
                        for col in temp_cols[:7]:
                            input_data[col] = st.number_input(f"{col}", value=data[col].mean())
                    
                    with col3:
                        st.subheader("Temperatures (T8-T14)")
                        for col in temp_cols[7:]:
                            input_data[col] = st.number_input(f"{col}", value=data[col].mean())
                    
                    # Submit button
                    submit_button = st.form_submit_button("Predict")
            
            else:  # Sample from Dataset
                # Let user select a random sample
                sample_idx = st.selectbox("Select a sample from the dataset", range(len(data)))
                
                # Get sample data
                sample = data.iloc[sample_idx]
                
                # Display sample
                st.dataframe(pd.DataFrame(sample).T)
                
                # Create input data
                for col in feature_names:
                    input_data[col] = sample[col]
                
                # Submit button
                submit_button = st.button("Predict")
            
            if submit_button:
                st.subheader("Prediction Result")
                
                # Try API first, fall back to local prediction
                result = predict_with_api(input_data)
                if result is None:
                    st.warning("API not available, falling back to local prediction")
                    result = predict_locally(input_data)
                
                if result is not None and result['status'] == 'success':
                    # Display prediction
                    st.success(f"Predicted Ethanol Concentration: **{result['prediction']:.5f}**")
                    st.info(f"Model Used: {result['model_used']}")
                    
                    # Create gauge chart for visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Define gauge range
                    min_val = max(0, data['Ethanol concentration'].min())
                    max_val = min(1, data['Ethanol concentration'].max())
                    
                    # Create gauge chart
                    pred_val = result['prediction']
                    
                    # Normalize the prediction value to a 0-1 range for the gauge
                    norm_val = (pred_val - min_val) / (max_val - min_val)
                    norm_val = max(0, min(1, norm_val))  # Ensure it's in [0,1]
                    
                    # Create a horizontal gauge
                    ax.barh(0, norm_val, height=0.5, color='green')
                    ax.barh(0, 1, height=0.5, color='lightgray', alpha=0.3)
                    
                    # Add markers
                    for i in np.linspace(0, 1, 11):
                        ax.plot([i, i], [-0.25, 0.25], 'k-', lw=1)
                        val = min_val + i * (max_val - min_val)
                        ax.text(i, -0.35, f'{val:.2f}', ha='center', va='center', fontsize=8)
                    
                    # Add prediction marker
                    ax.plot([norm_val, norm_val], [-0.25, 0.25], 'r-', lw=2)
                    ax.text(norm_val, 0.35, f'{pred_val:.5f}', ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='red')
                    
                    # Set axes properties
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_title('Predicted Ethanol Concentration')
                    
                    st.pyplot(fig)
                else:
                    st.error("Prediction failed. Please check your input values.")

if __name__ == "__main__":
    # This will be executed when the app is run directly
    pass
