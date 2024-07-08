import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Direct path to the data file
data_path = 'D:/laptop_price_prediction/data/laptop_prices.csv'

# Load the dataset
df = pd.read_csv(data_path)

# Handling missing values
df.fillna(0, inplace=True)

# Convert 'Price' to string, replace commas, and then convert to integer
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(int)

# Convert memory size columns to integers
df['ram_gb'] = df['ram_gb'].str.replace(' GB', '').astype(int)
df['ssd'] = df['ssd'].str.replace(' GB', '').astype(int)
df['hdd'] = df['hdd'].str.replace(' GB', '').astype(int)

# Convert categorical columns to numerical
categorical_columns = [
    'brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
    'ram_type', 'os', 'os_bit', 'graphic_card_gb', 'weight', 'warranty',
    'Touchscreen', 'msoffice', 'rating'
]
# Use one-hot encoding for categorical variables
df = pd.get_dummies(df, columns=categorical_columns)
# Define features and target
X = df.drop(columns=['Price'])
y = df['Price']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')
# Plot the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
# Display the distribution of errors
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.xlabel('Prediction Error')
plt.title('Distribution of Prediction Errors')
plt.show()