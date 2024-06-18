****Recommondation Sysytem****

**PROBLEM STATEMENT**

As Netflix's content library continues to rapidly expand, subscribers face increasing difficulty in discovering new titles that align with their personal interests and tastes. The sheer volume of available movies and TV shows makes manual browsing an inefficient and frustrating process, as users must sift through vast catalogs to identify relevant options amidst nuanced preferences.
To address this challenge, there is a growing need for an accurate and personalized recommendation system that can effectively match users with enjoyment-optimized viewing options based on a specific movie or TV show they have already watched and enjoyed. By leveraging a user's positive experience with a particular title, the system can analyze its characteristics and identify other similar titles the user is likely to appreciate.
However, developing such a robust recommendation engine requires access to proprietary data on user viewing histories, content preferences, and behavioral patterns – information that streaming platforms like Netflix closely guard. This lack of transparency presents barriers to external entities aiming to innovate in this space, underscoring the value of a recommendation system tailored specifically to the Netflix platform and its unique dataset.

**MODELS USED**
The models suggested by professor, such as regression and classification, are designed for supervised learning tasks where the data is labeled. However, our problem falls under the category of unsupervised learning, as the dataset we are working with is unlabeled. To effectively address this clustering problem, we have chosen to explore the following algorithms:
1.K-Means Clustering.
2.Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH).
3.Mean-shift Clustering.
4.Gaussian Mixture Model.
5.Agglomerative Clustering
6.Density Based Spatial Clustering of Applications with Noise (DBSCAN).
To develop an effective movie recommendation system for Netflix's vast library, we employed the above-mentioned unsupervised clustering algorithms to group similar titles together based on their inherent features. By leveraging a diverse set of clustering methodologies, our system can effectively identify natural groupings of movies/shows that share commonalities in genres, ratings, descriptions, and other relevant attributes. This clustering foundation enables personalized recommendations by matching a user's previously enjoyed title to other titles within the same cluster, maximizing the likelihood of suggesting content aligned with their preferences.
