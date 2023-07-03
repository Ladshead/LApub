import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
import seaborn as sns

# Simulated sensor data
x_values = [1, 2, 3, 4, 5]  # Replace with your sensor data for X-axis
y_values = [2, 4, 6, 8, 10]  # Replace with your sensor data for Y-axis
z_values = [5, 10, 15, 20, 25]  # Replace with your sensor data for Z-axis

# Convert the lists to NumPy arrays
x = np.array(x_values).reshape(-1, 1)
y = np.array(y_values).reshape(-1, 1)
z = np.array(z_values).reshape(-1, 1)

# Create the IsolationForest object
isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, max_samples='auto')

# Fit the IsolationForest object to the data
isolation_forest.fit(x)

# Predict the anomaly score for each data point
anomaly_scores = isolation_forest.decision_function(x)

# Set the threshold for classifying outliers
outlier_threshold = -0.5

# Identify outliers based on the threshold
outliers = anomaly_scores < outlier_threshold

# Create a DataFrame with the X, Y, and Z values
df = pd.DataFrame({'X': x_values, 'Y': y_values, 'Z': z_values, 'Outlier': outliers})

# Print the number of identified outliers
print('Number of outliers:', df['Outlier'].sum())
print('Anomaly scores:', anomaly_scores)

# Plot the identified outliers in the scatter plot
sns.scatterplot(data=df, x='X', y='Y', hue='Outlier', palette=['green', 'red'], size='Z', sizes=(50, 200), alpha=0.1)

# Specify the new save location and filename
save_path = "C:/SCIMIA/IsolationForest_outliers.png"
plt.savefig(save_path)
