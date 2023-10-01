import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window):
    data['RollingMean'] = data['Close'].rolling(window=window).mean()
    data['UpperBand'] = data['RollingMean'] + 2 * data['Close'].rolling(window=window).std()
    data['LowerBand'] = data['RollingMean'] - 2 * data['Close'].rolling(window=window).std()
    return data

# Function to calculate momentum
def calculate_momentum(data, window):
    data['Momentum'] = data['Close'].diff(window)
    return data

# Function to preprocess a CSV file
def preprocess_csv(file_path):
    data = pd.read_csv(file_path)
    data = calculate_bollinger_bands(data, window=20)
    data = calculate_momentum(data, window=5)
    return data

# Load and preprocess data from multiple CSV files
data_folder = 'data'  # Change to the path of your data folder
all_data = []

for root, _, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            processed_data = preprocess_csv(file_path)
            all_data.append(processed_data)

# Concatenate data from all files
combined_data = pd.concat(all_data)

# Drop rows with missing values
combined_data = combined_data.dropna()

# Split data into features (X) and target (Y)
X = combined_data[['UpperBand', 'LowerBand', 'Momentum']]
Y = combined_data['Close']

# Split data into training (70%) and testing (30%)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(trainX, trainY)

# Make predictions on the test data
test_predictions = model.predict(testX)

# Calculate and print RMSE (Root Mean Squared Error) for evaluation
rmse = math.sqrt(mean_squared_error(testY, test_predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")
