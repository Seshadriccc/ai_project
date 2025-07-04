import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
from q_optimizer import quantum_inspired_optimize

def train_model(csv_path):
    """Train XGBoost model with quantum-optimized features"""
    print(" Loading dataset...")
    df = pd.read_csv(csv_path)
    

    for col in ['Category', 'name']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    

    y = df['target']
    features = [col for col in df.columns if col != 'target']
    X = df[features]
    
    print(f" Dataset shape: {X.shape}")
    print(f" Features: {features}")
    

    hyperparams = [
        {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.15},
        {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.2},
        {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},
    ]
    

    (selected_features, best_params), quantum_score = quantum_inspired_optimize(features, hyperparams)
    
    print(f" Quantum-selected features: {selected_features}")
    print(f" Optimal hyperparameters: {best_params}")
    

    X_selected = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f" Model Performance:")
    print(f"   MSE: {mse:.2f}")
    print(f"   R²: {r2:.4f}")
    

    os.makedirs('models', exist_ok=True)
    joblib.dump((model, selected_features), 'models/xgb_model.pkl')
    
    print(" Model trained and saved!")
    return model, selected_features

if __name__ == "__main__":
    train_model("data/processed/processed_data.csv")