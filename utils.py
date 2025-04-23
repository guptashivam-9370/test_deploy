import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

def create_feature_importance_plot(model, feature_names, model_name):
    """
    Create feature importance plot for a given model
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        Figure with feature importance plot
    """
    plt.figure(figsize=(10, 6))
    
    # Linear models have coef_ attribute
    if hasattr(model, 'coef_'):
        # For linear models, coefficients can be positive or negative
        importances = model.coef_
        
        # Sort features by absolute importance
        indices = np.argsort(np.abs(importances))
        
        # Create a bar chart with positive and negative coefficients
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        
    # Tree-based models have feature_importances_ attribute
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)
        
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    
    # For SVR, we can't directly get feature importances
    else:
        plt.text(0.5, 0.5, "Feature importance not available for this model", 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance for {model_name}')
    plt.tight_layout()
    
    return plt.gcf()

def plot_correlation_matrix(df):
    """
    Create correlation matrix plot for numeric features
    
    Args:
        df: DataFrame with features
        
    Returns:
        Figure with correlation matrix plot
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    
    # Limit to at most 20 features for performance
    if len(numeric_df.columns) > 20:
        # Keep target column and most important features
        if 'Ethanol concentration' in numeric_df.columns:
            # Calculate correlation with target
            target = 'Ethanol concentration'
            correlations = numeric_df.corr()[target].abs().sort_values(ascending=False)
            # Keep target and top 19 most correlated features
            columns_to_keep = [target] + correlations.index[1:20].tolist()
            numeric_df = numeric_df[columns_to_keep]
        else:
            # Just take the first 20 columns
            numeric_df = numeric_df.iloc[:, :20]
    
    # Figure size based on number of features
    n_features = len(numeric_df.columns)
    figsize = (min(12, n_features*0.6), min(10, n_features*0.5))
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create heatmap - reduce annotations for large matrices
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Determine if annotations should be shown based on matrix size
    show_annot = n_features <= 15  # Only show annotations for smaller matrices
    
    sns.heatmap(corr, mask=mask, annot=show_annot, fmt=".2f" if show_annot else "", 
                cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    return plt.gcf()

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """
    Create scatter plot of actual vs predicted values
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model
        
    Returns:
        Figure with actual vs predicted plot
    """
    plt.figure(figsize=(8, 6))
    
    # Plot the scatter points
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # Plot the ideal line (y=x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Ethanol Concentration')
    plt.ylabel('Predicted Ethanol Concentration')
    plt.title(f'Actual vs Predicted Values - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_distributions(df):
    """
    Create distribution plots for features
    
    Args:
        df: DataFrame with features
        
    Returns:
        Figure with distribution plots
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    
    # Limit to at most 15 features for performance
    if len(numeric_df.columns) > 15:
        # Keep target column and most important features
        if 'Ethanol concentration' in numeric_df.columns:
            # Calculate correlation with target
            target = 'Ethanol concentration'
            correlations = numeric_df.corr()[target].abs().sort_values(ascending=False)
            # Keep target and top 14 most correlated features
            columns_to_keep = [target] + correlations.index[1:15].tolist()
            numeric_df = numeric_df[columns_to_keep]
        else:
            # Just take the first 15 columns
            numeric_df = numeric_df.iloc[:, :15]
    
    # Calculate number of rows and columns for subplots
    n_features = len(numeric_df.columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten()
    
    # Plot distributions - use faster matplotlib instead of seaborn for large datasets
    for i, col in enumerate(numeric_df.columns):
        # Use faster histogram for large datasets
        if len(df) > 1000:
            axes[i].hist(numeric_df[col], bins=30, alpha=0.7)
        else:
            sns.histplot(numeric_df[col], kde=True, ax=axes[i])
        
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig

def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """
    Generate a link to download the dataframe as a CSV file
    
    Args:
        df: DataFrame to download
        filename: Name of the download file
        text: Text to display for the download link
        
    Returns:
        HTML string with download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def create_model_comparison_plot(metrics_dict, metric_name='R2'):
    """
    Create bar chart comparing models based on a specific metric
    
    Args:
        metrics_dict: Dictionary with model metrics
        metric_name: Name of the metric to compare
        
    Returns:
        Figure with model comparison plot
    """
    plt.figure(figsize=(10, 6))
    
    # Map the display metric name to the actual key in the dictionary
    metric_key = 'R2' if metric_name == 'R²' else metric_name
    
    # Extract metric values for each model
    models = list(metrics_dict.keys())
    values = [metrics_dict[model][metric_key] for model in models]
    
    # Create bar chart
    bars = plt.bar(models, values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.title(f'Model Comparison - {metric_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()
