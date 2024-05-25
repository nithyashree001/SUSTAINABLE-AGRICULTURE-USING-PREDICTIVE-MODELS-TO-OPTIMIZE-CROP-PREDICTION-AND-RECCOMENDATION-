from __future__ import print_function
import csv
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
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

head = df.head()
print(head)

tail = df.tail()
print(tail)

size = df.size
print(size)

shape = df.shape
print(shape)

columns = df.columns
print(columns)

#missingvalues
miss = df.isnull().sum()
print(miss)

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_df = df[numeric_columns]
correlation_matrix = numeric_df.corr()

# Plotting the heatmap
dataplot = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  # 'coolwarm' is an example colormap
plt.show()

# Set Seaborn style
sns.set_style("darkgrid")

# Identify numerical columns
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

# Plot distribution of each numerical feature
plt.figure(figsize=(15, len(numerical_columns) * 1))
for idx, feature in enumerate(numerical_columns, 1):
	plt.subplot(len(numerical_columns), 1, idx)
	sns.histplot(df[feature], kde=True)
	plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")

# Adjust layout and show plots
plt.tight_layout()
plt.show()




