import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def perform_kmeans_clustering(input_file, personality_columns=None, max_clusters=10):
    """
    Perform K-means clustering on country personality scores
    
    Parameters:
    - input_file (str): Path to the CSV file with country personality averages
    - personality_columns (list, optional): Columns to use for clustering
    - max_clusters (int): Maximum number of clusters to evaluate
    
    Returns:
    - dict: Clustering results including optimal clusters, labels, and visualization data
    """
    # Default personality columns if not specified
    if personality_columns is None:
        personality_columns = [
            'agreeable_score', 
            'extraversion_score', 
            'openness_score', 
            'conscientiousness_score', 
            'neuroticism_score'
        ]
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Validate columns
    for col in personality_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file")
    
    # Prepare the data for clustering
    X = df[personality_columns]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using elbow method and silhouette score
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.title('Elbow Curve')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    
    # Plot silhouette scores
    plt.subplot(1,2,2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig('clustering_evaluation.png')
    plt.close()
    
    # Determine optimal number of clusters (highest silhouette score)
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Perform final clustering with optimal k
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_optimal.fit_predict(X_scaled)
    
    # Add cluster labels to the original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Save clustered data
    df_with_clusters.to_csv('country_personality_clusters.csv', index=False)
    
    # Cluster centers (in original scale)
    cluster_centers = scaler.inverse_transform(kmeans_optimal.cluster_centers_)
    cluster_centers_df = pd.DataFrame(
        cluster_centers, 
        columns=personality_columns
    )
    
    return {
        'optimal_clusters': optimal_k,
        'clustered_data': df_with_clusters,
        'cluster_centers': cluster_centers_df,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }

# Example usage
if __name__ == "__main__":
    try:
        # Personality columns to use for clustering
        personality_columns = [
            'agreeable_score', 
            'extraversion_score', 
            'openness_score', 
            'conscientiousness_score', 
            'neuroticism_score'
        ]
        
        # Perform K-means clustering
        clustering_results = perform_kmeans_clustering(
            'country_personality_averages.csv',
            personality_columns=personality_columns,
            max_clusters=10
        )
        
        print("Optimal Number of Clusters:", clustering_results['optimal_clusters'])
        
        print("\nCluster Centers:")
        print(clustering_results['cluster_centers'])
        
        print("\nClustered Data Preview:")
        print(clustering_results['clustered_data'].head())
        
        print("\nInertias:", clustering_results['inertias'])
        print("Silhouette Scores:", clustering_results['silhouette_scores'])
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Utility function for visualizing clusters
def visualize_clusters(clustered_data, personality_columns):
    """
    Create a parallel coordinates plot to visualize clusters
    
    Parameters:
    - clustered_data (pandas.DataFrame): Dataframe with cluster labels
    - personality_columns (list): Columns used for clustering
    """
    import plotly.graph_objs as go
    import plotly.offline as pyo
    
    # Normalize the data for visualization
    scaler = StandardScaler()
    normalized_data = clustered_data.copy()
    normalized_data[personality_columns] = scaler.fit_transform(normalized_data[personality_columns])
    
    # Create parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = clustered_data['Cluster'],
                        colorscale = 'Viridis',
                        showscale = True),
            dimensions = [dict(range = [normalized_data[col].min(), normalized_data[col].max()],
                               label = col, 
                               values = normalized_data[col]) for col in personality_columns]
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Country Personality Clusters',
        plot_bgcolor = 'white',
        paper_bgcolor = 'white'
    )
    
    # Save the plot
    pyo.plot(fig, filename='country_personality_clusters_parallel.html', auto_open=False)