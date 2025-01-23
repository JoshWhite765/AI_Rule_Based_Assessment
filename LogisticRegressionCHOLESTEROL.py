
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

data = pd.read_csv('processed.hungarian.csv')

data.columns = ["Age", "Sex", "ChestPType", "Blood_Pressure", "Cholesterol", "BloodSugar", "electrocardio", "MaxHeartRate", "Exang", "Oldpeak", "Slope", "NumMajorVessles","Thal", "Heart_Disease_Risk"]

# Replace all question mark values with Nan
data.replace('?', np.nan, inplace=True)

# ChatGPT replace NaN with mean 
num_cols = ['Cholesterol'] 
num_imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean
data[num_cols] = num_imputer.fit_transform(data[num_cols])


X = data[['Cholesterol']] # Feature 
y = data['Heart_Disease_Risk'] # Target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression classifier
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

#Calculate the metrics 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

precision = precision_score(y_test, y_pred)
print(f'Precision: {precision * 100:.2f}%')

recall = recall_score(y_test, y_pred)
print(f'Recall: {recall * 100:.2f}%')

f1 = f1_score(y_test, y_pred) 
print(f'F1-Score: {f1 * 100:.2f}%')


# ChatGPT to plot the regression line
plt.figure(figsize=(8,6))
plt.scatter(data['Cholesterol'], data['Heart_Disease_Risk'], c=data['Heart_Disease_Risk'], cmap='coolwarm', edgecolors='k', alpha=0.7)
x_vals = np.linspace(data['Cholesterol'].min(), data['Cholesterol'].max(), 100)
y_vals = log_reg.predict_proba(x_vals.reshape(-1, 1))[:, 1]  # Probability of heart disease risk (1)
plt.plot(x_vals, y_vals, color='red', label='Logistic Regression')
plt.title('Cholesterol Vs Angiographic heart disease risk (Logistic Regression)')
plt.xlabel('Cholesterol')
plt.ylabel('Heart Disease Risk Probability')
plt.colorbar(label='Angiographic heart disease status (0 = No, 1 = Yes)')
plt.legend()
plt.show()



