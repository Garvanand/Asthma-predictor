import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
df = pd.read_csv("asthma_detection.csv")

# Print data info
print("\nData Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Prepare features and target
print("\nPreparing features and target...")
X = df.drop('Asthma', axis=1)
y = df['Asthma']

# Print feature names
print("\nFeature names:")
print(X.columns.tolist())

# Split the data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Train the model with GridSearchCV
print("\nTraining model with GridSearchCV...")
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the model
train_score = best_model.score(X_train_balanced, y_train_balanced)
test_score = best_model.score(X_test, y_test)

print(f"\nTraining accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Additional metrics
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
print("\nSaving model...")
dump(best_model, 'rf_asthma_model_prediction.pkl')
print("Model saved successfully!") 