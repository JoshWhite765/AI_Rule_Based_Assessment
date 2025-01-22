import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
data = pd.read_csv('C:/Users/josh/OneDrive - Bath Spa University/GitHub/AI_Rule_Based_Assessment/processed.hungarian.csv')

#define data columns
data.columns = ["Age", "Sex", "ChestPType", "Blood_Pressure", "Cholesterol", "BloodSugar", "electrocardio", 
                "MaxHeartRate", "Exang", "Oldpeak", "Slope", "NumMajorVessles", "Thal", "Heart_Disease_Risk"]

# Set features and targets
X = data[['Cholesterol', 'Age']] 
y = data['Heart_Disease_Risk']   

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#Train KNN
knn = KNeighborsClassifier(n_neighbors=20) # set the value of k to 5 
knn.fit(X_train, y_train)


# Predict for visualization
y_pred = knn.predict(X_test)

#Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

#Calculate the precision
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision * 100:.2f}%')

#Calculate the recall
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall * 100:.2f}%')

#Calculate the F1-Score
f1 = f1_score(y_test, y_pred) 
print(f'F1-Score: {f1 * 100:.2f}%')

# ChatGPT to Plot values
plt.figure(figsize=(16, 6))
scatter = plt.scatter(X_test['Cholesterol'], X_test['Age'], c=y_pred, cmap='viridis', edgecolor='k', s=50)
plt.colorbar(scatter, label='Predicted Risk')
plt.title('KNN Classification of Heart Disease Risk')
plt.xlabel('Cholesterol')
plt.ylabel('Age')
plt.show()
