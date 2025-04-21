import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor

class ModelTrainer:
    """
    Class for training and evaluating regression models
    """
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
            'Ridge Regression': Ridge(alpha=0.1, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, epsilon=0.1),
            'CatBoost Regressor': CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, 
                                                   verbose=False, random_state=42)
        }
        self.trained_models = {}
        self.model_metrics = {}
        
    def train_models(self, X_train, y_train, cv=5):
        """
        Train all models and perform cross-validation
        
        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with model metrics
        """
        metrics = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation scores
            cv_mse = -cross_val_score(model, X_train, y_train, 
                                     scoring='neg_mean_squared_error', cv=cv)
            cv_r2 = cross_val_score(model, X_train, y_train, 
                                   scoring='r2', cv=cv)
            
            # Save trained model
            self.trained_models[name] = model
            
            # Save metrics
            metrics[name] = {
                'CV_MSE_mean': np.mean(cv_mse),
                'CV_MSE_std': np.std(cv_mse),
                'CV_R2_mean': np.mean(cv_r2),
                'CV_R2_std': np.std(cv_r2)
            }
        
        self.model_metrics = metrics
        return metrics
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate trained models on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with model evaluation metrics
        """
        eval_metrics = {}
        
        for name, model in self.trained_models.items():
            print(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save metrics
            eval_metrics[name] = {
                'MSE': mse,
                'R2': r2,
                'Predictions': y_pred
            }
            
            # Update overall metrics
            self.model_metrics[name].update(eval_metrics[name])
        
        return eval_metrics
    
    def get_best_model(self, metric='R2'):
        """
        Get the best model based on a specific metric
        
        Args:
            metric: Metric to use for model selection ('R2' or 'MSE')
            
        Returns:
            Name of the best model and the model object
        """
        if not self.model_metrics:
            return None, None
        
        if metric == 'R2':
            # Higher R2 is better
            best_model_name = max(self.model_metrics, key=lambda k: self.model_metrics[k]['R2'])
        elif metric == 'MSE':
            # Lower MSE is better
            best_model_name = min(self.model_metrics, key=lambda k: self.model_metrics[k]['MSE'])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_model_name, self.trained_models[best_model_name]
    
    def save_models(self, directory='saved_models'):
        """
        Save trained models to disk
        
        Args:
            directory: Directory to save models
            
        Returns:
            Dictionary with paths to saved models
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        saved_paths = {}
        
        for name, model in self.trained_models.items():
            # Create a valid filename
            filename = f"{name.replace(' ', '_').lower()}.joblib"
            path = os.path.join(directory, filename)
            
            # Save model
            joblib.dump(model, path)
            saved_paths[name] = path
        
        return saved_paths
    
    def load_model(self, model_path):
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        return joblib.load(model_path)
