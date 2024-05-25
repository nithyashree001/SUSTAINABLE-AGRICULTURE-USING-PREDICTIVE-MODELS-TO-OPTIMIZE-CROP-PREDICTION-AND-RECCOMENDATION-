from __future__ import print_function
import csv
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import tree
df = pd.read_csv('INDIANCROP_DATA.csv')

state_mapping = state_mapping = {'Andaman and Nicobar': 0,'Andhra Pradesh': 1,'Assam': 2,'Chattisgarh': 3,'Goa': 4,'Gujarat': 5,'Haryana': 6,
                                 'Himachal Pradesh': 7,'Jammu and Kashmir': 8,'Karnataka': 9,'Kerala': 10,'Madhya Pradesh': 11,'Maharashtra': 12,
                                 'Manipur': 13,'Meghalaya': 14,'Nagaland': 15,'Odisha': 16,'Pondicherry': 17,'Punjab': 18,'Rajasthan': 19,
                                 'Tamil Nadu': 20,'Telangana': 21,'Tripura': 22,'Uttar Pradesh': 23,'Uttrakhand': 24,'West Bengal': 25}
df['STATE'] = df['STATE'].map(state_mapping)
df = df.drop_duplicates()
print("\nUpdated DataFrame:")
print(df)
print(df.dtypes)

target = df['CROP']
labels = df['CROP']
acc = []
model = []

features = df[['N_SOIL', 'P_SOIL','K_SOIL','TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL','CROP_PRICE','STATE']]
features_encoded = pd.get_dummies(features)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

RF = RandomForestClassifier(n_estimators=20, random_state=42 ,max_depth = 5)

# Train the Random Forest classifier on the training data
RF.fit(Xtrain, Ytrain)

# Make predictions on the test data
predicted_values = RF.predict(Xtest)

# Calculate and print accuracy
accuracy = metrics.accuracy_score(Ytest, predicted_values)
print("Random Forest's Accuracy is: {:.2f}%".format(accuracy * 100))

# Append accuracy and model name for further analysis
acc.append(accuracy)
model.append('RF')

# Print classification report
print("Classification Report:")
print(classification_report(Ytest, predicted_values))


DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)

# Train the Decision Tree on the training data
DecisionTree.fit(Xtrain, Ytrain)

# Make predictions on the test data
predicted_values = DecisionTree.predict(Xtest)

# Calculate and print accuracy
accuracy = metrics.accuracy_score(Ytest, predicted_values)
print("Decision Tree's Accuracy is: {:.2f}%".format(accuracy * 100))

# Append accuracy and model name for further analysis
acc.append(accuracy)
model.append('Decision Tree')

# Print classification report
print("Classification Report:")
print(classification_report(Ytest, predicted_values))


LogReg = LogisticRegression(random_state=2)

# Train the Logistic Regression model on the training data
LogReg.fit(Xtrain, Ytrain)

# Make predictions on the test data
predicted_values = LogReg.predict(Xtest)

# Calculate and print accuracy
accuracy = metrics.accuracy_score(Ytest, predicted_values)
print("Logistic Regression's Accuracy is: {:.2f}%".format(accuracy * 100))

# Append accuracy and model name for further analysis
acc.append(accuracy)
model.append('Logistic Regression')

# Print classification report
print("Classification Report:")
print(classification_report(Ytest, predicted_values))



cv_scores = cross_val_score(RF, features, target, cv=5)
print("Cross-Validation Scores:", cv_scores)

conf_matrix = confusion_matrix(Ytest, predicted_values)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Confusion Matrix of RF')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

feature_importances = RF.feature_importances_

# Print or visualize feature importances
print("Feature Importances:")
for feature, importance in zip(Xtrain.columns, feature_importances):
    print(f"{feature}: {importance}")

# Assuming 'Xtrain' is your training data with columns
# and 'feature_importances' is the array of feature importances

# Sort features based on importance
sorted_idx = feature_importances.argsort()

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), Xtrain.columns[sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importances")
plt.show()

N_SOIL = int(input("Enter N_SOIL value in %: "))
P_SOIL = int(input("Enter P_SOIL value in %: "))
K_SOIL = int(input("Enter K_SOIL value in %: "))
TEMPERATURE = float(input("Enter TEMPERATURE value in Celsius: "))
HUMIDITY = float(input("Enter HUMIDITY value in %: "))
ph = float(input("Enter ph value: "))
RAINFALL = float(input("Enter RAINFALL value in mm: "))
CROP_PRICE = int(input("Enter CROP_PRICE value in rs : "))
STATE_NAME = input("Enter STATE name: ")

# Map the state name to the corresponding number
STATE = state_mapping.get(STATE_NAME)

# Check if the entered state name is valid
if STATE is None:
    print("Invalid state name!")
else:
    # Create a numpy array with user input
    data = np.array([[N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL, CROP_PRICE, STATE]])

    # Make predictions
    prediction = RF.predict(data)

    # Print the prediction
    print("The suitable crop is:", prediction)






