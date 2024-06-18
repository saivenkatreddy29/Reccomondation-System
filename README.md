# Reccomondation-System
PROBLEM STATEMENT
As Netflix's content library continues to rapidly expand, subscribers face increasing difficulty in discovering new titles that align with their personal interests and tastes. The sheer volume of available movies and TV shows makes manual browsing an inefficient and frustrating process, as users must sift through vast catalogs to identify relevant options amidst nuanced preferences.
To address this challenge, there is a growing need for an accurate and personalized recommendation system that can effectively match users with enjoyment-optimized viewing options based on a specific movie or TV show they have already watched and enjoyed. By leveraging a user's positive experience with a particular title, the system can analyze its characteristics and identify other similar titles the user is likely to appreciate.
However, developing such a robust recommendation engine requires access to proprietary data on user viewing histories, content preferences, and behavioral patterns – information that streaming platforms like Netflix closely guard. This lack of transparency presents barriers to external entities aiming to innovate in this space, underscoring the value of a recommendation system tailored specifically to the Netflix platform and its unique dataset.
DATA SOURCES AND DETAILS ABOUT DATA
The dataset was compiled and published on Hugging Face by user 'hugginglearners' in November 2019.
The exact original sources of the Netflix catalog data are unclear, but it appears to have been web scraped or extracted from Netflix's public-facing applications/APIs around 2019.
Dataset Details :
Over 8,000 shows and movies available on Netflix
12 attributes per title:
oshow_id - unique ID for the title.
otype – says whether the content is a movie or a TV show.
otitle - name of title.
odirector - directors (for movies).
ocast - main actors/actresses.
ocountry - production country.
odate_added - date when added to Netflix.
orelease_year - year when title was released.
orating - MPAA rating.
oduration - duration in minutes (movies or episode).
olisted_in - genre(s).
odescription - short text description of the plot.
Data is a mix of categorical variables like genres and text fields like descriptions, along with continuous numeric data like duration and discrete data like years and dates.
The data provides a snapshot of the Netflix catalog in 2019 rather than longitudinal view.
Biases exist around greater representation of English-language content.
This dataset provides a valuable basis for analysis despite some limitations, enabling insights into the content types, availability by country, ratings, release timeframes etc. The inclusion of multiple data types allows for a multitude of analyses.
MODELS USED
The models suggested by professor, such as regression and classification, are designed for supervised learning tasks where the data is labeled. However, our problem falls under the category of unsupervised learning, as the dataset we are working with is unlabeled. To effectively address this clustering problem, we have chosen to explore the following algorithms:
1.K-Means Clustering.
2.Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH).
3.Mean-shift Clustering.
4.Gaussian Mixture Model.
5.Agglomerative Clustering
6.Density Based Spatial Clustering of Applications with Noise (DBSCAN).
To develop an effective movie recommendation system for Netflix's vast library, we employed the above-mentioned unsupervised clustering algorithms to group similar titles together based on their inherent features. By leveraging a diverse set of clustering methodologies, our system can effectively identify natural groupings of movies/shows that share commonalities in genres, ratings, descriptions, and other relevant attributes. This clustering foundation enables personalized recommendations by matching a user's previously enjoyed title to other titles within the same cluster, maximizing the likelihood of suggesting content aligned with their preferences.
1.K-Means Clustering
To group similar movies together and enable personalized recommendations, we employed the K-Means clustering algorithm from scikit-learn. This partitioning technique aims to divide the dataset into K distinct non-overlapping clusters, where each data point (movie) belongs to the cluster with the nearest mean (centroid).
Initially, we extracted the movie titles and features from the dataset. The features were then scaled using StandardScaler to ensure equal contribution from all dimensions during the clustering process.
Determining Optimal K:
 One crucial step in K-Means is determining the optimal number of clusters (K). We used the Elbow method, which calculates the sum of squared errors (SSE) for different values of K. The plot shows how the SSE decreases as K increases, with an "elbow" point at K=34, indicating the optimal trade-off between minimizing SSE and avoiding excessive clusters.

Model Training:
 	With the optimal K value determined, we trained the K-Means model using the scaled movie features. The algorithm iteratively assigned each movie to the nearest centroid and updated the centroids based on the new cluster assignments, until convergence.
Cluster Assignment: 
The trained model assigned a cluster label to each movie in the dataset, enabling us to group similar movies together based on their inherent features.
Recommendation Function: 
We implemented a function to recommend similar movies given a target movie title. This function operates as follows:
1.Locate the target movie in the dataset and extract its features and cluster assignment.
2.Identify all movies belonging to the same cluster as the target movie.
3.Calculate the cosine similarity between the target movie's features and all other movies within the cluster.
4.Return the top N most similar movies based on the cosine similarity scores.
 The output demonstrates the recommended movies for the input "Kota Factory," showcasing the algorithm's ability to suggest relevant titles based on shared attributes.

Hyper parameter Tuning:
We changed the value of k three times which are 22, 34 in which both the model with k = 34 performed well. We chose k=37 and moved forward. The evaluation metrics for k = 22 are in the below image and the ones for k = 37 are discussed below.

Evaluation Metrics:
 To assess the clustering performance, we calculated several evaluation metrics:
Silhouette Score: 0.44683242265294545 
The Silhouette Score measures how well each data point (movie) fits into its assigned cluster compared to other clusters. The score ranges from -1 to 1, with a higher value indicating that the data point is well-matched to its cluster and poorly matched to neighboring clusters.
In this case, the Silhouette Score of 0.446 is a moderately positive value, suggesting that the clustering is reasonably good, but there is room for improvement. A score closer to 1 would indicate that the movies are very well-clustered and separated from other clusters.
Calinski-Harabasz Index: 935.845623849385 
The Calinski-Harabasz Index is a ratio of the sum of between-cluster dispersion and inter-cluster dispersion for all clusters. A higher value indicates that the clusters are dense and well-separated from each other.
While there is no definitive interpretation of the Calinski-Harabasz Index value, a higher value is generally considered better. In this case, the value of 935.845 suggests that the clustering is reasonably good.
Davies-Bouldin Index: 0.8452928989425635 
The Davies-Bouldin Index is a measure of the average similarity between clusters, where lower values indicate better separation between clusters. A value of 0 would indicate that the clusters are completely separated, while higher values suggest that the clusters are closer together and potentially overlapping.
The obtained Davies-Bouldin Index of 0.845 is a moderately low value, indicating that the clusters are reasonably well-separated, but there is still some overlap or similarity between clusters.
Homogeneity: 0.3558867248431815 
Completeness: 1.0000000000000002 
V-measure: 0.5256266647739224 
The Homogeneity, Completeness, and V-measure metrics evaluate the quality of the clustering by comparing it to ground truth labels.
Visualization:
To gain insights into the clustering results, we used Principal Component Analysis (PCA) to reduce the feature dimensions to 2 for visualization purposes. The below Image displays a scatter plot of the movies, colored by their assigned cluster labels, revealing the natural groupings identified by the K-Means algorithm.

2.Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH)
We employed the BIRCH algorithm from scikit-learn to try out one more Clustering Algorithm. This hierarchical clustering technique is designed to efficiently handle large datasets by constructing a tree-like structure, making it well-suited for our extensive Netflix movie catalog.
Similar to the K-Means approach, we extracted the movie titles and features from the dataset and scaled the features using StandardScaler to ensure equal contribution across dimensions during the clustering process.
Model Training: 
The BIRCH algorithm was trained on the scaled movie features, with a branching factor of 50 and a threshold of 0.5. These hyperparameters control the size and compactness of the clusters, respectively. BIRCH builds a tree-like representation of the data, iteratively assigning movies to clusters and refining the cluster boundaries until convergence.
Cluster Assignment: 
After training, the BIRCH model assigned a cluster label to each movie in the dataset, enabling us to group similar titles together based on their intrinsic attributes.
Recommendation Function: 
Analogous to the K-Means implementation, we developed a function to recommend similar movies given a target movie title. The steps involved are:
1.Locate the target movie in the dataset and extract its features and cluster assignment.
2.Identify all movies belonging to the same cluster as the target movie.
3.Calculate the cosine similarity between the target movie's features and all other movies within the cluster.
4.Return the top N most similar movies based on the highest cosine similarity scores.

Hyperparameter Tuning:
We tried tuning the model with threshold of 0.5 and 0.8. In both the cases the model performed very similarly. Hence we moved on with threshold = 0.5.
Evaluation Metrics:
To assess the clustering performance, we calculated several evaluation metrics:
Silhouette Score: 0.8563415349789212 
The Silhouette Score measures how well each data point fits into its assigned cluster compared to other clusters. A higher value indicates better clustering, with 1 being the highest possible score. The obtained Silhouette Score of 0.856 suggests that the data points (movies) are well-matched to their respective clusters and well-separated from other clusters, indicating good clustering quality.
Calinski-Harabasz Index: 8338.35904429055

The Calinski-Harabasz Index measures the ratio of between-cluster dispersion and inter-cluster dispersion. A higher value indicates denser and well-separated clusters. The obtained value of 8338.359 is considered very high, suggesting that the clusters are dense and well-separated from each other, indicating excellent clustering performance.
Davies-Bouldin Index: 0.1402481325852651
The Davies-Bouldin Index measures the average similarity between clusters, with lower values indicating better separation. The obtained value of 0.140 is considered very low, suggesting that the clusters are well-separated and have minimal overlap or similarity between them.
Homogeneity: 0.6261554343809398
Completeness: 1.0000000000000002
V-measure: 0.7701052693272346
The Homogeneity, Completeness, and V-measure metrics evaluate the clustering quality by comparing it to ground truth labels. Homogeneity measures the extent to which each cluster contains only data points from a single ground truth class, while Completeness measures the extent to which all data points from a given ground truth class are assigned to the same cluster.
The obtained Homogeneity score of 0.626 and Completeness score of 1.0 suggest that the clustering has moderately high homogeneity (each cluster contains mostly data points from a single class) and perfect completeness (all data points from a given class are assigned to the same cluster). The V-measure, which is the harmonic mean of Homogeneity and Completeness, has a value of 0.770, indicating overall good clustering quality when compared to the ground truth labels
