"""
Script to train and save a housing price prediction model.
This uses the California housing dataset as an example.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Create the model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load the California housing dataset
print("Loading dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Display info about the dataset
print(f"Dataset shape: {X.shape}")
print(f"Features: {housing.feature_names}")

# Check for missing values
missing_values = X.isnull().sum()
print(f"Missing values: {missing_values.sum()}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Create pipelines for different models
print("Creating model pipelines...")
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

pipeline_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso())
])

# Set up grid search parameters
param_grid_ridge = {
    'model__alpha': [0.1, 1.0, 10.0]
}

param_grid_lasso = {
    'model__alpha': [0.01, 0.1, 1.0]
}

# Train and evaluate models
print("Training Linear Regression model...")
pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression - MSE: {mse_lr:.4f}, R²: {r2_lr:.4f}")

print("Performing GridSearch for Ridge model...")
grid_ridge = GridSearchCV(pipeline_ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
grid_ridge.fit(X_train, y_train)
print(f"Best Ridge parameters: {grid_ridge.best_params_}")
y_pred_ridge = grid_ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge - MSE: {mse_ridge:.4f}, R²: {r2_ridge:.4f}")

print("Performing GridSearch for Lasso model...")
grid_lasso = GridSearchCV(pipeline_lasso, param_grid_lasso, cv=5, scoring='neg_mean_squared_error')
grid_lasso.fit(X_train, y_train)
print(f"Best Lasso parameters: {grid_lasso.best_params_}")
y_pred_lasso = grid_lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso - MSE: {mse_lasso:.4f}, R²: {r2_lasso:.4f}")

# Choose the best model (lowest MSE)
models = {
    'Linear Regression': (pipeline_lr, mse_lr, r2_lr),
    'Ridge': (grid_ridge.best_estimator_, mse_ridge, r2_ridge),
    'Lasso': (grid_lasso.best_estimator_, mse_lasso, r2_lasso)
}

best_model_name = min(models, key=lambda k: models[k][1])
best_model, best_mse, best_r2 = models[best_model_name]

print(f"The best model is {best_model_name} with MSE: {best_mse:.4f} and R²: {best_r2:.4f}")

# Save the best model
model_filename = 'model/model.pkl'
joblib.dump(best_model, model_filename)
print(f"Best model saved as {model_filename}")

# Save feature names for use in the API
feature_names = X.columns.tolist()
with open('model/feature_names.txt', 'w') as f:
    f.write(','.join(feature_names))
print(f"Feature names saved to model/feature_names.txt: {feature_names}")

# Display coefficients or feature importance
if hasattr(best_model.named_steps['model'], 'coef_'):
    coefs = best_model.named_steps['model'].coef_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefs)})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    print("\nFeature coefficients:")
    print(feature_importance)