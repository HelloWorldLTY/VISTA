import numpy as np
import faiss
from scipy.sparse import csr_matrix

def knn_graph(data, n_neighbors=5):
    """
    Computes the k-nearest neighbor graph using faiss and returns a sparse matrix.
    
    Parameters:
    - data: ndarray, shape (n_samples, n_features)
        Input data points
    - k: int
        Number of neighbors to retrieve
        
    Returns:
    - sparse_matrix: csr_matrix, shape (n_samples, n_samples)
        Sparse matrix where each row represents a data point and the non-zero columns in that row 
        are the k-nearest neighbors. The values are the distances to the neighbors.
    """
    
    k = n_neighbors
    # Convert data to float32 (required by Faiss)
    data = data.astype(np.float32)
    
    # Build the Faiss index (using the L2 distance)
    d = data.shape[1]  # data dimensionality
    index = faiss.IndexFlatL2(d)
    index.add(data)
    
    # Query the index to get the k+1 nearest neighbors (k+1 because the point itself is included)
    distances, indices = index.search(data, k + 1)
    
    # Convert to sparse matrix format
    n_samples = data.shape[0]
    
    # Prepare data for CSR format
    row_indices = np.repeat(np.arange(n_samples), k)
    col_indices = indices[:, 1:].flatten()  # Exclude the first column (point itself)
    distance_data = np.sqrt(distances[:, 1:].flatten())  # Convert squared L2 distances to L2 distances
    
    # distance_data = (distance_data > 0)*1
    
    # Create the CSR matrix
    sparse_matrix = csr_matrix((distance_data, (row_indices, col_indices)), shape=(n_samples, n_samples))
    
    return sparse_matrix
