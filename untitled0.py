# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:02:57 2023

@author: dario
"""

#Individual project Dario Pijman
#Creating a Fraud detection model 

#First generate a dataset
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

legitimate_timestamps = pd.date_range(start='2023-01-01', end='2023-01-30', freq='H')
legitimate_transactions = pd.DataFrame({
    'timestamp': legitimate_timestamps,
    'amount': np.random.uniform(10, 500, size=len(legitimate_timestamps)),
    'hour': legitimate_timestamps.hour,
    'location': np.random.choice(['A', 'B', 'C'], size=len(legitimate_timestamps)),
    'fraud_label': 0  
})

fraudulent_timestamps = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 30, size=30), unit='D')
fraudulent_transactions = pd.DataFrame({
    'timestamp': fraudulent_timestamps,
    'amount': np.random.uniform(500, 5000, size=len(fraudulent_timestamps)),
    'hour': np.random.randint(0, 24, size=len(fraudulent_timestamps)),
    'location': np.random.choice(['A', 'B', 'C'], size=len(fraudulent_timestamps)), 
    'fraud_label': 1  
})

synthetic_dataset = pd.concat([legitimate_transactions, fraudulent_transactions], ignore_index=True)

synthetic_dataset = synthetic_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
csv_file_path = os.path.join(downloads_path, 'synthetic_fraud_dataset2.csv')

synthetic_dataset.to_csv(csv_file_path, index=False)

print("File saved to:", csv_file_path)


#Perform Isolation Forest Model and XGBoost Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

synthetic_dataset = pd.read_csv('synthetic_fraud_dataset2.csv')

synthetic_dataset['timestamp'] = pd.to_datetime(synthetic_dataset['timestamp'])
synthetic_dataset['day_of_week'] = synthetic_dataset['timestamp'].dt.dayofweek
synthetic_dataset['day_of_month'] = synthetic_dataset['timestamp'].dt.day
synthetic_dataset['week_of_year'] = synthetic_dataset['timestamp'].dt.isocalendar().week
synthetic_dataset['hour'] = synthetic_dataset['timestamp'].dt.hour
synthetic_dataset['minute'] = synthetic_dataset['timestamp'].dt.minute

label_encoder = LabelEncoder()
synthetic_dataset['location'] = label_encoder.fit_transform(synthetic_dataset['location'])

X = synthetic_dataset.drop(['timestamp', 'fraud_label'], axis=1)

y = synthetic_dataset['fraud_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_isolation_forest = IsolationForest(contamination=0.05, random_state=42)
model_isolation_forest.fit(X_train_scaled)

y_pred_isolation_forest = model_isolation_forest.predict(X_test_scaled)
y_pred_isolation_forest = np.where(y_pred_isolation_forest == -1, 1, 0)

print("Isolation Forest Model:")
print(classification_report(y_test, y_pred_isolation_forest))
print("Accuracy:", accuracy_score(y_test, y_pred_isolation_forest))

model_xgboost = XGBClassifier()
model_xgboost.fit(X_train_scaled, y_train)

y_pred_xgboost = model_xgboost.predict(X_test_scaled)

print("\nXGBoost Model:")
print(classification_report(y_test, y_pred_xgboost))
print("Accuracy:", accuracy_score(y_test, y_pred_xgboost))



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

synthetic_dataset = pd.read_csv('synthetic_fraud_dataset2.csv')

location_fraud_crosstab = pd.crosstab(synthetic_dataset['location'], synthetic_dataset['fraud_label'])

plt.figure(figsize=(8, 6))
sns.heatmap(location_fraud_crosstab, annot=True, fmt='d', cmap='Blues', cbar=True,
            annot_kws={"size": 12}, linewidths=.5,
            linecolor='lightgray', vmin=0, vmax=synthetic_dataset['fraud_label'].value_counts().max())

cbar = plt.gcf().axes[-1]
cbar.set_ylabel('Count', rotation=270, labelpad=15)

plt.xticks(ticks=[0.5, 1.5], labels=['No Fraud', 'Fraud'], rotation=0)
plt.yticks(rotation=0)

plt.title('Correlation between Location and Fraud')
plt.show()


location_fraud_percentage = location_fraud_crosstab.div(location_fraud_crosstab.sum(axis=1), axis=0) * 100

colors = ["#1f78b4", "#d62728"]

plt.figure(figsize=(10, 6))
location_fraud_percentage.plot(kind='bar', stacked=True, color=colors)
plt.title('Relative Proportion of Fraud by Location')
plt.xlabel('Location')
plt.ylabel('Percentage')

plt.legend(title='Fraud Label', labels=['No Fraud', 'Fraud'], loc='upper left', bbox_to_anchor=(1, 1))

plt.show()

import pandas as pd
from scipy.stats import chi2_contingency

synthetic_dataset = pd.read_csv('synthetic_fraud_dataset2.csv')

contingency_table = pd.crosstab(synthetic_dataset['location'], synthetic_dataset['fraud_label'])

chi2, p, _, _ = chi2_contingency(contingency_table)

print(f"Chi-squared value: {chi2}")
print(f"P-value: {p}")

if p < 0.05:
    print("The distribution of fraud across locations is statistically significant.")
else:
    print("The distribution of fraud across locations is not statistically significant.")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

synthetic_dataset = pd.read_csv('synthetic_fraud_dataset1.csv')

plt.figure(figsize=(10, 6))
sns.boxplot(x='fraud_label', y='amount', data=synthetic_dataset)
plt.title('Distribution of Amounts for Fraud and No Fraud Transactions')
plt.xlabel('Fraud Label')
plt.ylabel('Amount')
plt.show()

fraud_amounts = synthetic_dataset.loc[synthetic_dataset['fraud_label'] == 1, 'amount']
no_fraud_amounts = synthetic_dataset.loc[synthetic_dataset['fraud_label'] == 0, 'amount']

t_stat, p_value = ttest_ind(fraud_amounts, no_fraud_amounts, equal_var=False)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("The difference in mean amounts between fraud and no fraud transactions is statistically significant.")
else:
    print("There is no statistically significant difference in mean amounts between fraud and no fraud transactions.")


import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

synthetic_dataset = pd.read_csv('synthetic_fraud_dataset2.csv')

features_for_clustering = synthetic_dataset[['amount', 'hour', 'fraud_label', 'location']].copy()

label_encoder = LabelEncoder()
features_for_clustering['location'] = label_encoder.fit_transform(features_for_clustering['location'])

scaler = StandardScaler()
features_for_clustering_scaled = scaler.fit_transform(features_for_clustering)

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
synthetic_dataset['cluster'] = kmeans.fit_predict(features_for_clustering_scaled)

sns.pairplot(data=synthetic_dataset, hue='cluster', palette='viridis', markers='o')
plt.suptitle('Pair Plots for Clusters')
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

synthetic_dataset = pd.read_csv('synthetic_fraud_dataset1.csv')

X = synthetic_dataset[['amount', 'hour', 'location']]
y = synthetic_dataset['fraud_label']

label_encoder = LabelEncoder()
X['location'] = label_encoder.fit_transform(X['location'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_naive_bayes = GaussianNB()
model_naive_bayes.fit(X_train_scaled, y_train)

y_pred_naive_bayes = model_naive_bayes.predict(X_test_scaled)

print("Naive Bayes Model:")
print(classification_report(y_test, y_pred_naive_bayes))
print("Accuracy:", accuracy_score(y_test, y_pred_naive_bayes))


print("Naive Bayes Model:")
print(classification_report(y_test, y_pred_naive_bayes))
print("Accuracy:", accuracy_score(y_test, y_pred_naive_bayes))















































