# Clustering Assignment Solutions

## Conceptual Questions

### 1. What is unsupervised learning in the context of machine learning?
Unsupervised learning is a type of machine learning where algorithms learn patterns from unlabeled data without predefined target variables. The goal is to discover hidden structures, groupings, or relationships within the data. Clustering is a primary application of unsupervised learning.

### 2. How does K-Means clustering algorithm work?
K-Means algorithm:
1. Initialize k random centroids
2. Assign each data point to the nearest centroid
3. Calculate new centroids as the mean of assigned points
4. Repeat steps 2-3 until convergence (centroids don't change significantly)
5. Produces k clusters with minimized within-cluster variance

### 3. Explain the concept of a dendrogram in hierarchical clustering
A dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows:
- Leaf nodes represent individual data points
- Internal nodes represent merged clusters
- Height of connections indicates the distance/dissimilarity at which clusters merge
- Used to determine optimal number of clusters by cutting the tree at appropriate height

### 4. What is the main difference between K-Means and Hierarchical Clustering?
| Aspect | K-Means | Hierarchical |
|--------|---------|------------|
| Approach | Partitioning | Agglomerative/Divisive |
| Number of clusters | Must specify k in advance | Can be determined from dendrogram |
| Output | Flat partition | Nested hierarchy |
| Scalability | Better for large datasets | Computationally expensive |
| Time Complexity | O(n*k*i*d) | O(n²) to O(n³) |

### 5. What are the advantages of DBSCAN over K-Means?
- Can discover clusters of arbitrary shapes (not just spherical)
- No need to specify number of clusters beforehand
- Identifies and handles noise points/outliers
- Performs well on datasets with varying cluster densities
- Works with density-connected regions rather than distance-based centers

### 6. When would you use Silhouette Score in clustering?
Use Silhouette Score to:
- Evaluate quality of clustering results
- Determine optimal number of clusters
- Validate if data points are well-matched to their assigned clusters
- Compare different clustering algorithms objectively
- Range: [-1, 1]; higher values indicate better-defined clusters

### 7. What are the limitations of Hierarchical Clustering?
- Computationally expensive: O(n²) to O(n³) time complexity
- Sensitive to noise and outliers
- Cannot undo previous merges (greedy approach)
- Difficult to interpret dendrograms for large datasets
- May not work well with high-dimensional data
- Once merged, clusters cannot be separated

### 8. Why is feature scaling important in clustering algorithms like K-Means?
- K-Means uses Euclidean distance; unscaled features with larger ranges dominate
- Ensures all features contribute equally to distance calculation
- Prevents features with larger units from biasing cluster formation
- Without scaling, features like age (0-100) would dominate over income ratios (0-1)
- Improves algorithm convergence and clustering quality

### 9. How does DBSCAN identify noise points?
DBSCAN identifies noise points as:
- Points that have fewer than `min_samples` neighbors within `eps` radius
- Not in any cluster and not reachable from core points
- Exist in sparse regions with low density
- Labeled separately, allowing discovery of true outliers

### 10. Define inertia in the context of K-Means
Inertia is the sum of squared Euclidean distances of each point to its nearest centroid:
```
Inertia = Σ(distance from point to centroid)²
```
- Lower inertia indicates more compact, cohesive clusters
- Always decreases as k increases
- Used in elbow method to find optimal clusters

### 11. What is the elbow method in K-Means clustering?
The elbow method identifies optimal k by:
1. Running K-Means for k = 1 to n
2. Computing inertia for each k
3. Plotting inertia vs. k
4. Identifying the "elbow" point where inertia decrease slows
5. The elbow k is typically the optimal cluster count

### 12. Describe the concept of "density" in DBSCAN
Density in DBSCAN refers to:
- Number of points within an `eps` radius around a point
- A point is core if it has ≥ `min_samples` neighbors within `eps`
- Dense regions form clusters; sparse regions form noise
- DBSCAN groups density-connected core points
- Different regions can have different densities

### 13. Can hierarchical clustering be used on categorical data?
Yes, but with modifications:
- Use distance metrics suited for categorical data (Hamming distance, Gower distance)
- Standard Euclidean distance doesn't apply
- Requires encoding categorical variables appropriately
- Can use hierarchical clustering with non-numeric data if proper distance functions defined
- Alternative: use methods like DIANA for categorical data

### 14. What does a negative Silhouette Score indicate?
A negative Silhouette Score indicates:
- Data point is closer to other clusters than its assigned cluster
- Point is likely misclassified or assigned to wrong cluster
- Cluster structure is weak or overlapping
- Suggests poor clustering quality for that point
- Range [-1, 0): worse than random cluster assignment

### 15. Explain the term "linkage criteria" in hierarchical clustering
Linkage criteria determine how distance between clusters is calculated:
- **Single Linkage**: Distance = min distance between any two points
- **Complete Linkage**: Distance = max distance between any two points
- **Average Linkage**: Distance = average distance between all point pairs
- **Ward Linkage**: Minimizes variance within merged clusters
- Choice affects dendrogram shape and final clustering results

### 16. Why might K-Means clustering perform poorly on data with varying cluster sizes or densities?
- K-Means assumes spherical clusters of similar sizes
- Uses Euclidean distance from centroids; biased toward larger clusters
- Small or dense clusters may be split; sparse clusters may merge
- Centroid-based approach favors compact, uniform-density clusters
- Fails on non-convex or elongated cluster shapes

### 17. What are the core parameters in DBSCAN, and how do they influence clustering?
**eps (epsilon)**: Radius of neighborhood
- Larger eps: more points in clusters, fewer noise points
- Smaller eps: smaller clusters, more noise points

**min_samples**: Minimum neighbors for core point
- Larger value: stricter core point definition, more noise
- Smaller value: looser definition, larger clusters

**metric**: Distance measure (default: Euclidean)
- Affects how similarity is calculated

### 18. How does K-Means++ improve upon standard K-Means initialization?
K-Means++ initialization:
- Selects first centroid randomly
- Each subsequent centroid chosen with probability proportional to squared distance from nearest centroid
- Spreads initial centroids far apart
- Converges faster and finds better local optima
- Reduces sensitivity to poor random initialization
- Yields more consistent, high-quality clusters

### 19. What is agglomerative clustering?
Agglomerative clustering is a bottom-up hierarchical approach:
- Starts with each point as separate cluster
- Iteratively merges closest/most similar clusters
- Produces dendrogram showing merge history
- Requires distance metric and linkage criteria
- Number of clusters determined by cutting dendrogram
- Common linkage methods: single, complete, average, Ward

### 20. What makes Silhouette Score a better metric than just inertia for model evaluation?
- **Inertia limitation**: Always decreases with increasing k; doesn't reveal clustering quality
- **Silhouette advantages**:
  - Measures both cluster cohesion and separation
  - Accounts for inter-cluster distances
  - Provides interpretable metric [-1, 1]
  - Independent of k
  - Can identify poorly clustered points
  - Better for comparing different algorithms and parameters

---

## Theoretical Questions (with Python Code)

### 1. Generate synthetic data with 4 centers using make_blobs and apply K-Means clustering. Visualize using a scatter plot.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=0.6)
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, edgecolors='black', linewidth=2, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering with 4 Centers')
plt.legend()
plt.show()
```

### 2. Load the Iris dataset and use Agglomerative Clustering to group the data into 3 clusters. Display the first 10 predicted labels.

```python
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()
X = iris.data
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clustering.fit_predict(X)

print("First 10 predicted labels:", labels[:10])
```

### 3. Generate synthetic data using make_moons and apply DBSCAN. Highlight outliers in the plot.

```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.15, min_samples=5)
labels = dbscan.fit_predict(X)

plt.scatter(X[labels != -1, 0], X[labels != -1, 1], c=labels[labels != -1], 
            cmap='viridis', alpha=0.6, edgecolors='k', label='Clusters')
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], c='red', marker='x', 
            s=100, linewidth=2, label='Outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN on Make Moons Data')
plt.legend()
plt.show()
```

### 4. Load the Wine dataset and apply K-Means clustering after standardizing the features. Print the size of each cluster.

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

wine = load_wine()
X = wine.data
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

unique, counts = np.unique(labels, return_counts=True)
for cluster, size in zip(unique, counts):
    print(f"Cluster {cluster}: {size} samples")
```

### 5. Use make_circles to generate synthetic data and cluster it using DBSCAN. Plot the result.

```python
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, _ = make_circles(n_samples=300, noise=0.05, random_state=42, factor=0.5)
X = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.15, min_samples=5)
labels = dbscan.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN on Make Circles Data')
plt.show()
```

### 6. Load the Breast Cancer dataset, apply MinMaxScaler, and use K-Means with 2 clusters. Output the cluster centroids.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

cancer = load_breast_cancer()
X = cancer.data
X_scaled = MinMaxScaler().fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

print("Cluster Centroids:\n", kmeans.cluster_centers_)
```

### 7. Generate synthetic data using make_blobs with varying cluster standard deviations and cluster with DBSCAN.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=[0.5, 1.0, 1.5], random_state=42)
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.title('DBSCAN on Blobs with Varying Standard Deviations')
plt.show()
```

### 8. Load the Digits dataset, reduce it to 2D using PCA, and visualize clusters from K-Means.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X_pca)

pca_centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, edgecolors='k')
plt.scatter(pca_centers[:, 0], pca_centers[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidth=2)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means on Digits Dataset (PCA Reduced)')
plt.show()
```

### 9. Create synthetic data using make_blobs and evaluate silhouette scores for k = 2 to 5. Display as a bar chart.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

silhouette_scores = []
k_range = range(2, 6)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.bar(k_range, silhouette_scores, color='steelblue', edgecolor='black')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k Values')
plt.xticks(k_range)
plt.show()
```

### 10. Load the Iris dataset and use hierarchical clustering to group data. Plot a dendrogram with average linkage.

```python
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

linkage_matrix = linkage(X, method='average')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, leaf_rotation=90)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram (Average Linkage)')
plt.show()
```

### 11. Generate synthetic data with overlapping clusters using make_blobs, then apply K-Means and visualize with decision boundaries.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, edgecolors='black', linewidth=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means with Decision Boundaries')
plt.show()
```

### 12. Load the Digits dataset and apply DBSCAN after reducing dimensions with t-SNE. Visualize the results.

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
X_tsne = StandardScaler().fit_transform(X_tsne)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_tsne)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('DBSCAN on Digits Dataset (t-SNE Reduced)')
plt.show()
```

### 13. Generate synthetic data using make_blobs and apply Agglomerative Clustering with complete linkage. Plot the result.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='complete')
labels = agg_clustering.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Agglomerative Clustering (Complete Linkage)')
plt.show()
```

### 14. Load the Breast Cancer dataset and compare inertia values for K = 2 to 6 using K-Means. Show results in a line plot.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X = cancer.data
X_scaled = StandardScaler().fit_transform(X)

inertias = []
k_range = range(2, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(k_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)
plt.show()
```

### 15. Generate synthetic concentric circles using make_circles and cluster using Agglomerative Clustering with single linkage.

```python
from sklearn.datasets import make_circles
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

X, _ = make_circles(n_samples=300, noise=0.05, random_state=42, factor=0.5)

agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='single')
labels = agg_clustering.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Agglomerative Clustering - Single Linkage (Circles)')
plt.show()
```

### 16. Use the Wine dataset, apply DBSCAN after scaling the data, and count the number of clusters (excluding noise).

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

wine = load_wine()
X = wine.data
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {list(labels).count(-1)}")
```

### 17. Generate synthetic data with make_blobs and apply KMeans. Then plot the cluster centers on top of the data points.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k', label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='*', s=500, edgecolors='black', linewidth=2, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering with Centroids')
plt.legend()
plt.show()
```

### 18. Load the Iris dataset, cluster with DBSCAN, and print how many samples were identified as noise.

```python
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

noise_count = list(labels).count(-1)
print(f"Number of noise points: {noise_count}")
```

### 19. Generate synthetic non-linearly separable data using make_moons, apply K-Means, and visualize the clustering result.

```python
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, edgecolors='black', linewidth=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means on Make Moons (Non-linear Data)')
plt.show()
```

### 20. Load the Digits dataset, apply PCA to reduce to 3 components, then use KMeans and visualize with a 3D scatter plot.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X_pca)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='tab10', alpha=0.6, edgecolors='k')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('K-Means on Digits Dataset (3D PCA)')
plt.show()
```

---

## Practical Questions (with Complete Solutions)

### 1. Generate synthetic blobs with 5 centers and apply KMeans. Then use silhouette_score to evaluate the clustering.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

X, _ = make_blobs(n_samples=300, centers=5, random_state=42)

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)

score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.4f}")
```

### 2. Load the Breast Cancer dataset, reduce dimensionality using PCA, and apply Agglomerative Clustering. Visualize in 2D.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X = cancer.data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = agg_clustering.fit_predict(X_pca)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Agglomerative Clustering on Breast Cancer Dataset (PCA)')
plt.show()
```

### 3. Generate noisy circular data using make_circles and visualize clustering results from KMeans and DBSCAN side-by-side.

```python
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, _ = make_circles(n_samples=300, noise=0.05, random_state=42, factor=0.5)
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6, edgecolors='k')
axes[0].set_title('K-Means on Make Circles')

axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6, edgecolors='k')
axes[1].set_title('DBSCAN on Make Circles')

plt.tight_layout()
plt.show()
```

### 4. Load the Iris dataset and plot the Silhouette Coefficient for each sample after KMeans clustering.

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

silhouette_vals = silhouette_samples(X, labels)

y_lower = 10
for i in range(3):
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    plt.fill_betweenx(range(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7)
    y_lower = y_upper + 10

plt.xlabel('Silhouette Coefficient')
plt.ylabel('Cluster Label')
plt.title('Silhouette Coefficients for Iris Dataset')
plt.show()
```

### 5. Generate synthetic data using make_blobs and apply Agglomerative Clustering with 'average' linkage. Visualize clusters.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='average')
labels = agg_clustering.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Agglomerative Clustering with Average Linkage')
plt.show()
```

### 6. Load the Wine dataset, apply KMeans, and visualize the cluster assignments in a seaborn pairplot (first 4 features).

```python
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wine = load_wine()
X = wine.data[:, :4]  # First 4 features
feature_names = wine.feature_names[:4]

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

df = pd.DataFrame(X, columns=feature_names)
df['Cluster'] = labels

sns.pairplot(df, hue='Cluster', palette='viridis')
plt.suptitle('Wine Dataset - Cluster Assignments (First 4 Features)', y=1.00)
plt.show()
```

### 7. Generate noisy blobs using make_blobs and use DBSCAN to identify both clusters and noise points. Print the count.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
X = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.4, min_samples=5)
labels = dbscan.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
```

### 8. Load the Digits dataset, reduce dimensions using t-SNE, then apply Agglomerative Clustering and plot the clusters.

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

agg_clustering = AgglomerativeClustering(n_clusters=10, linkage='ward')
labels = agg_clustering.fit_predict(X_tsne)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.6, edgecolors='k')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Agglomerative Clustering on Digits Dataset (t-SNE Reduced)')
plt.show()
```

---

## Key Takeaways

- **K-Means**: Fast, efficient, but assumes spherical clusters
- **Hierarchical**: Flexible linkage options, builds dendrogram, slower
- **DBSCAN**: Discovers arbitrary shapes, handles noise naturally
- **Feature Scaling**: Essential for distance-based algorithms
- **Silhouette Score**: Best metric for evaluating cluster quality
- **Elbow Method**: Use inertia to find optimal k
- **Noise Handling**: Only DBSCAN explicitly identifies outliers
- **Practical Approach**: Use Silhouette Score + domain knowledge for best results

