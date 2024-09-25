import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the TF-IDF matrix from a CSV file
def load_tfidf_matrix(file_name):
    # Load the matrix
    tfidf_matrix = pd.read_csv(file_name, index_col=0)  # Assume the first column is the index (document names)
    return tfidf_matrix

# Step 2: Visualize with PCA
def visualize_with_pca(tfidf_matrix):
    # Reduce dimensionality with PCA
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(tfidf_matrix.values)  # Use .values to get the matrix as a NumPy array

    # Create a DataFrame for visualization
    pca_df = pd.DataFrame(pca_results, columns=['Component 1', 'Component 2'])
    pca_df['Document'] = tfidf_matrix.index  # Use the index for document names

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['Component 1'], pca_df['Component 2'])

    # Annotate points with document names
    for i, txt in enumerate(pca_df['Document']):
        plt.annotate(txt, (pca_df['Component 1'][i], pca_df['Component 2'][i]), fontsize=9)

    plt.title('PCA Visualization of TF-IDF Matrix')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid()
    plt.show()

# Main execution
file_name = 'TF-IDF.csv'  # Replace with your CSV file name
tfidf_matrix = load_tfidf_matrix(file_name)
visualize_with_pca(tfidf_matrix)
