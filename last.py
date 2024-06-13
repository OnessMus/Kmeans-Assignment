import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans


data = pd.read_csv('OnlineRetail.csv', encoding = 'latin 1')
#print(data.head())
#print(data.info())
#print(data.describe())
#print(data.isnull().sum())
#print(data.shape)
#print(data.columns)
'''percent_missing = 100 * data.isnull().sum() / len(data)
rounded_percent_missing = round(percent_missing, 2)
print(rounded_percent_missing)'''

data = data.dropna()
#print(data.shape)

data['CustomerID'] = data['CustomerID'].astype(str)

#Data Preparation
#creating new column
data['Amount'] = data['Quantity']*data['UnitPrice']
#print(data.info())

monetary = data.groupby('CustomerID')['Amount'].sum()
#print(monetary)

most_sold = data.groupby('Description')['Quantity'].sum()
most_sold = most_sold.idxmax()
#print(most_sold)

best_region = data.groupby('Country')['Quantity'].sum()
best_region = best_region.idxmax()
#print(best_region)

monetary = monetary.reset_index()
#print(monetary)

frequently_sold =  data.groupby('Description')['InvoiceNo'].count()
frequently_sold = frequently_sold.idxmax()
#print(frequently_sold)

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate']) #,format='%d/%m/%Y %H:%M', errors='coerce')
#print("After formatting: \n", data['InvoiceDate'])

max_date = max(data['InvoiceDate'])
#print("The last transaction was on:", max_date)

min_date = min(data['InvoiceDate'])
#print("The earliest transaction was on:", min_date)

total_duration = max_date - min_date
#print("The total duration of transactions:", total_duration)

from datetime import timedelta
thirty_days_ago = max_date - timedelta(days=30)
#print("Thirty days ago from the latest transaction:", thirty_days_ago)

transactions_within_period = data[data['InvoiceDate'] >= thirty_days_ago]
total_sales = transactions_within_period['Amount'].sum()
#print("Total sales for the last thirty days:", total_sales ,'dollars')

# Remove rows with missing CustomerID
data_clean = data.dropna(subset=['CustomerID'])

# Create TotalPrice feature
data_clean['TotalPrice'] = data_clean['Quantity'] * data_clean['UnitPrice']

# Select relevant features for clustering
features = data_clean[['Quantity', 'TotalPrice']]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Use the Elbow Method to determine the optimal number of clusters
wcss = []
num_clusters = range(1, 11)

for i in num_clusters:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(num_clusters, wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(num_clusters)
plt.grid(True)
#plt.show()

# Optimal number of clusters (from the Elbow Method, let's assume it's 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Calculate WCSS for the chosen number of clusters
final_wcss = kmeans.inertia_
print(f'Final WCSS for {optimal_clusters} clusters: {final_wcss:.4f}')

# Calculate Silhouette Score
silhouette_avg = silhouette_score(scaled_features, cluster_labels)
print(f'Silhouette Score: {silhouette_avg:.4f}')
