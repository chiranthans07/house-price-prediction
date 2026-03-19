#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:20:16 2026

@author: chiranthansateesh
"""

# --------------------------------------------
# House Price Prediction using Linear Regression
# --------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("Housing.csv")

print("Dataset Loaded Successfully\n")
print(data.head())

# Select features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display predictions
print("\nPredicted Prices:")
print(y_pred[:5])

# Visualization (Actual vs Predicted)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()