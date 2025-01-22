
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

data = pd.read_csv('processed.hungarian.csv')

#Define data columns
data.columns = ["Age", "Sex", "ChestPType", "Blood_Pressure", "Cholesterol", "BloodSugar", "electrocardio", "MaxHeartRate", "Exang", "Oldpeak", "Slope", "NumMajorVessles", "Thal", "Heart_Disease_Risk"]

data.replace('?', np.nan, inplace=True)

#ChatGPT replace Nan with mean
num_cols = ['Cholesterol']
num_imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean
data[num_cols] = num_imputer.fit_transform(data[num_cols])

# Create binary target: 1 if age > 50, 0 otherwise
data['Age_Above_50'] = (data['Age'] > 50).astype(int)

# Define features (Cholesterol) and target for (Age Above/Below 50)
X = data[['Cholesterol']]
y = data['Age_Above_50']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression classifier
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

# chatGPT to plot 
plt.figure(figsize=(8,6))
plt.scatter(data['Cholesterol'], data['Age_Above_50'], c=data['Age_Above_50'], cmap='coolwarm', edgecolors='k', alpha=0.7)
x_vals = np.linspace(data['Cholesterol'].min(), data['Cholesterol'].max(), 100)
y_vals = log_reg.predict_proba(x_vals.reshape(-1, 1))[:, 1]  # Probability of Age > 50
plt.plot(x_vals, y_vals, color='red', label='Logistic Regression')
plt.title('Cholesterol vs Probability of Age Above 50 (Logistic Regression)')
plt.xlabel('Cholesterol')
plt.ylabel('Probability of Age Above 50')
plt.colorbar(label='Age Group Probability')
plt.legend()
plt.show()
