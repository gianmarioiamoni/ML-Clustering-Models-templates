# K-Means Clustering

# Identify clusters of features
#
# We create dependant variable in such a way that each of the values
# of this dependant variable are actually the classes of the variable

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#
# Importing the dataset
#
# We have only the matrix of features
# There is not dependant variable in the dataset
# The goal is to identify clusters. 1st column is not relevant
# We keep only columns 3,4 to have a 2 dimensional plot
# In real case all column but the 1st are relevant
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
#
# We use the KMeans class of the scikit-learn library
# We use a for loop to run the algorithm with a different number
# of clusters, from 1 to 10. 
# For each number of clusters, we calulate WCSS, which is the sum of
# sqare distances between each observation point of the cluster
# and its central width.
# this will be reported on the Y axis in the graph of the elbow method.
from sklearn.cluster import KMeans
# List that will be populated with the successive WCSS values
wcss = []
# Run the algorithm with a different number of clusters
for i in range(1, 11):
    # create a kmeans object, which represent a KMeans algorithm,
    # as instance of KMeans class
    #
    # n_clusters: number of clusters we want to try
    # init: initialize the k-means++ algorithm to avoid the Randome Initialization Trap
    # random_state: random seed
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    # Train the algorithm.
    # We need only the features as argument
    kmeans.fit(X)
    # Add the computes WCSS value to the list.
    # inertia_: attribute of the kmeans object storing the WCSS value
    wcss.append(kmeans.inertia_)

# Plot the WCSS values
#
# We use the plot function of the matplotlib library
# to plot the WCSS values
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
#
# From the Elbow method, we can see that the optimum number of clusters
# is 5.
# We use the same code we used in the loop of the elbow method
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
# Create the dependant variable by using fit_predict() method for the training
y_kmeans = kmeans.fit_predict(X)

# We can see all the differetn clusters which belongs each customer
# Each value of the list corrsponds the the cluster to which the 
# customer at that position in the dataset belongs to
print(y_kmeans)

# Visualising the clusters
#
# We create a scatter plot per each cluster
# We use a different color for each cluster
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')  
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

# Plot the centroid of each cluster
# We use the cluster_centers_ attribute of the kmeans object
# Its a 2D array containing in the rows all the different centroids
# and in the columns their coordinates
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()