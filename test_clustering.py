import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import skewnorm

# Import data
df = pd.read_csv('problem_filter.csv', index_col=False)
df['vnd/m2'] = df['price_vnd']/df['square']
df.head(5)

# Visualize geo-location data
plt.figure(figsize=(10,10))
sns.scatterplot(x='longitude', y='latitude', data=df)
plt.title('House Distribution')
plt.savefig('data_scatter.png', transparent=True)

# Set variables
X = np.array(df[['longitude', 'latitude']].astype(float))
weights = np.array(df['vnd/m2'])

# Visualize and compare non-weighted vs weighted results
plt.figure(figsize=(18, 18))
for K in range(3,6):
    # Plot non-weighted clustering
    kmeans = KMeans(n_clusters=K, random_state=0, max_iter=1000)
    kmeansclus_nw = kmeans.fit(X)
    predicted_kmeans_nw = kmeans.predict(X)
    centers_nw = kmeansclus_nw.cluster_centers_
    
    plt.subplots()
    plt.scatter(X[:, 0], X[:, 1], c=predicted_kmeans_nw, s=10, cmap='tab20')
    plt.scatter(centers_nw[:, 0], centers_nw[:, 1], c='black', s=200, alpha=0.5)
    plt.title('Distribution with non-Weighted K-Means')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    if K == 4:
        plt.savefig('nw.png', transparent=True)
    
    # Plot weighted clustering
    kmeans = KMeans(n_clusters=K, random_state=0, max_iter=1000)
    wt_kmeansclus = kmeans.fit(X,sample_weight = weights)
    predicted_kmeans = kmeans.predict(X, sample_weight = weights)
    centers = wt_kmeansclus.cluster_centers_

    plt.subplots()
    plt.scatter(X[:,0], X[:,1], c=wt_kmeansclus.labels_.astype(float), s=10, cmap='tab20b')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title('Distribution with Weighted K-Means')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    if K == 4:
        plt.savefig('wt.png', transparent=True)