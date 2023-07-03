import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from sklearn.ensemble import IsolationForest
import seaborn as sns
import re

# This has taken the file from my downloads, to be used on the hardware this would need to be edited
path = str(Path.home() / "Downloads")
os.chdir(path)
with open('Magnetometer-Temperature_Serial data.txt', 'r') as file:
    contents = file.read()

pattern = r"X (-?\d+), Y (-?\d+), Z (-?\d+)"
matches = re.findall(pattern, contents)

# Extract X, Y, Z values into separate lists
x_values = []
y_values = []
z_values = []
for match in matches:
    x_values.append(int(match[0]))
    y_values.append(int(match[1]))
    z_values.append(int(match[2]))

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
save_path = "C:/IsolationForest_outliers.png"
plt.savefig(save_path)
