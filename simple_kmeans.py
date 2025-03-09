# comprehensive_kmeans.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs, fetch_olivetti_faces
from sklearn.decomposition import PCA
import cv2
import os

# Create folders if they don't exist
os.makedirs('images', exist_ok=True)
os.makedirs('analysis', exist_ok=True)


def simple_kmeans_example():
    """
    Simple K-means clustering example using generated data.
    """
    # Generate sample data with 4 clusters
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

    # Apply K-means clustering with k=4
    kmeans = KMeans(n_clusters=4, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot each cluster with a different color
    colors = ['red', 'blue', 'green', 'purple']
    for i in range(4):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1],
                    s=50, c=colors[i], label=f'Cluster {i + 1}')

    # Plot the centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=200, c='black', marker='X', label='Centroids')

    plt.title('K-means Clustering on Generated Data (k=4)', fontsize=14)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/kmeans_simple_example.png', dpi=300)
    plt.close()


def iris_kmeans_example():
    """
    K-means clustering example using the Iris dataset.
    """
    # Load the iris dataset
    iris = load_iris()
    X = iris.data

    # Apply K-means clustering with k=3 (known number of species)
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # For visualization, we'll use PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot each cluster with a different color
    colors = ['red', 'blue', 'green']
    for i in range(3):
        plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1],
                    s=50, c=colors[i], label=f'Cluster {i + 1}')

    # Plot the centroids (projected to PCA space)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                s=200, c='black', marker='X', label='Centroids')

    plt.title('K-means Clustering on Iris Dataset (k=3)', fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/kmeans_iris_example.png', dpi=300)
    plt.close()


def faces_kmeans_example():
    """
    K-means clustering example using the Olivetti faces dataset.
    """
    # Load the faces dataset
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data

    # Apply PCA to reduce dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Apply K-means clustering with k=5
    kmeans = KMeans(n_clusters=5, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot each cluster with a different color
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i in range(5):
        plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1],
                    s=50, c=colors[i], label=f'Cluster {i + 1}')

    # Plot the centroids (projected to PCA space)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                s=200, c='black', marker='X', label='Centroids')

    plt.title('K-means Clustering on Olivetti Faces Dataset (k=5)', fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/kmeans_faces_example.png', dpi=300)
    plt.close()

    # Display example faces from each cluster
    plt.figure(figsize=(15, 8))

    for i in range(5):
        # Get indices of faces in this cluster
        cluster_indices = np.where(y_kmeans == i)[0]

        # Display up to 4 example faces from this cluster
        for j in range(min(4, len(cluster_indices))):
            if j < len(cluster_indices):
                plt.subplot(5, 4, i * 4 + j + 1)
                plt.imshow(faces.images[cluster_indices[j]], cmap='gray')
                plt.title(f'Cluster {i + 1}')
                plt.axis('off')

    plt.tight_layout()
    plt.savefig('images/kmeans_faces_examples.png', dpi=300)
    plt.close()


def image_compression_example(image_path='sample_image.png', k_values=[2, 5, 10, 20]):
    """
    Use K-means for image compression with different values of k.

    Parameters:
    -----------
    image_path : str
        Path to the input image file
    k_values : list
        List of k values to use for compression
    """
    try:
        # Read the image
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        # Convert from BGR to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width, channels = img.shape

        # Reshape the image to a 2D array of pixels
        pixels = img.reshape(-1, 3)

        # Create a figure to show original and compressed images
        plt.figure(figsize=(15, 10))

        # Plot the original image
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title('Original Image', fontsize=12)
        plt.axis('off')

        # Get original image file size
        with open(image_path, 'rb') as f:
            original_size = len(f.read())

        print(f"Original image size: {original_size / 1024:.2f} KB")

        # Compress the image with different k values
        for i, k in enumerate(k_values):
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(pixels)

            # Replace each pixel with its closest cluster center
            compressed = kmeans.cluster_centers_[kmeans.labels_]

            # Reshape back to original image shape
            compressed_img = compressed.reshape(height, width, channels)
            compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)

            # Plot the compressed image
            plt.subplot(2, 3, i + 2)
            plt.imshow(compressed_img)
            plt.title(f'K={k} Colors', fontsize=12)
            plt.axis('off')

            # Calculate compression ratio
            # Save compressed image to measure its size
            compressed_path = f'analysis/compressed_k{k}_{os.path.basename(image_path)}'
            plt.imsave(compressed_path, compressed_img)

            with open(compressed_path, 'rb') as f:
                compressed_size = len(f.read())

            compression_ratio = original_size / compressed_size
            print(f"K={k} colors: {compressed_size / 1024:.2f} KB, Compression ratio: {compression_ratio:.2f}x")

            # Save the compressed image for comparison
            cv2.imwrite(compressed_path, cv2.cvtColor(compressed_img, cv2.COLOR_RGB2BGR))

        plt.tight_layout()
        plt.savefig('analysis/image_compression_comparison.png', dpi=300)
        plt.close()

        print("Image compression analysis completed. Check the 'analysis' folder.")

    except Exception as e:
        print(f"Error in image compression: {e}")


def color_analysis_example(image_path='sample_image.png', k=8):
    """
    Use K-means to extract and analyze dominant colors in an image.

    Parameters:
    -----------
    image_path : str
        Path to the input image file
    k : int
        Number of color clusters to extract
    """
    try:
        # Read the image
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Reshape the image to a 2D array of pixels
        pixels = img.reshape(-1, 3)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)

        # Get the dominant colors
        colors = kmeans.cluster_centers_

        # Convert to integer values
        colors = colors.astype(int)

        # Count pixels per cluster
        labels_count = np.bincount(kmeans.labels_)

        # Calculate percentage of each color
        percentages = labels_count / len(pixels) * 100

        # Sort colors by their frequency (most frequent first)
        sorted_indices = np.argsort(-percentages)
        percentages = percentages[sorted_indices]
        colors = colors[sorted_indices]

        # Create a color palette visualization
        plt.figure(figsize=(12, 8))

        # Plot the original image
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.title('Original Image', fontsize=14)
        plt.axis('off')

        # Plot the color palette
        plt.subplot(2, 1, 2)
        for i in range(len(colors)):
            plt.bar(i, percentages[i], color=colors[i] / 255)
            plt.text(i, percentages[i] + 1, f"{percentages[i]:.1f}%",
                     ha='center', fontsize=12)

        plt.title('Dominant Colors', fontsize=14)
        plt.xticks(range(len(colors)), [f"Color {i + 1}" for i in range(len(colors))])
        plt.xlabel('Color')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, max(percentages) * 1.2)

        plt.tight_layout()
        plt.savefig('analysis/color_analysis.png', dpi=300)
        plt.close()

        # Create color swatches
        plt.figure(figsize=(12, 2))
        for i in range(len(colors)):
            plt.subplot(1, len(colors), i + 1)
            plt.axhspan(0, 1, color=colors[i] / 255)
            plt.text(0.5, 0.5, f"{percentages[i]:.1f}%",
                     ha='center', va='center', color='white' if sum(colors[i]) < 380 else 'black',
                     fontsize=12, fontweight='bold')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('analysis/color_swatches.png', dpi=300)
        plt.close()

        # Print the RGB values of dominant colors
        print("\nDominant Colors Analysis:")
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            print(f"Color {i + 1}: RGB{tuple(color)} - {percentage:.1f}%")

        print("Color analysis completed. Check the 'analysis' folder.")

    except Exception as e:
        print(f"Error in color analysis: {e}")


def segmentation_example(image_path='sample_image.png', k=3):
    """
    Use K-means for image segmentation.

    Parameters:
    -----------
    image_path : str
        Path to the input image file
    k : int
        Number of segments to create
    """
    try:
        # Read the image
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Reshape the image to a 2D array of pixels
        pixels = img.reshape(-1, 3)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pixels)

        # Create segmented image
        segmented = kmeans.cluster_centers_[labels].reshape(img.shape)
        segmented = np.clip(segmented, 0, 255).astype(np.uint8)

        # Create mask images for each segment
        masks = []
        for i in range(k):
            mask = (labels == i).reshape(img.shape[0], img.shape[1])
            masks.append(mask)

        # Plot the results
        plt.figure(figsize=(15, 8))

        # Plot the original image
        plt.subplot(2, k + 1, 1)
        plt.imshow(img)
        plt.title('Original Image', fontsize=12)
        plt.axis('off')

        # Plot the segmented image
        plt.subplot(2, k + 1, 2)
        plt.imshow(segmented)
        plt.title('Segmented Image', fontsize=12)
        plt.axis('off')

        # Plot each segment
        for i in range(k):
            # Create a colored segment
            segment = np.zeros_like(img)
            segment[masks[i], :] = img[masks[i], :]

            plt.subplot(2, k + 1, i + 3)
            plt.imshow(segment)
            plt.title(f'Segment {i + 1}', fontsize=12)
            plt.axis('off')

            # Save individual segments
            plt.imsave(f'analysis/segment_{i + 1}.png', segment)

        plt.tight_layout()
        plt.savefig('analysis/image_segmentation.png', dpi=300)
        plt.close()

        print("Image segmentation completed. Check the 'analysis' folder.")

    except Exception as e:
        print(f"Error in image segmentation: {e}")


def create_sample_image(output_path='sample_image.png'):
    """
    Create a sample image if none is provided.
    """
    try:
        # Check if the file already exists
        if os.path.exists(output_path):
            print(f"Using existing image at {output_path}")
            return

        print(f"Creating sample image at {output_path}")

        # Create a sample image with shapes and colors
        width, height = 800, 600
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw a blue circle
        cv2.circle(img, (200, 200), 100, (255, 0, 0), -1)

        # Draw a green rectangle
        cv2.rectangle(img, (400, 100), (700, 300), (0, 255, 0), -1)

        # Draw a red triangle
        pts = np.array([[300, 400], [200, 500], [400, 500]], np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 255))

        # Draw a yellow star
        center = (600, 400)
        pts = []
        for i in range(5):
            # Outer point (star tip)
            angle = np.pi / 2 + i * 2 * np.pi / 5
            x = int(center[0] + 100 * np.cos(angle))
            y = int(center[1] + 100 * np.sin(angle))
            pts.append([x, y])

            # Inner point
            angle = np.pi / 2 + (i + 0.5) * 2 * np.pi / 5
            x = int(center[0] + 40 * np.cos(angle))
            y = int(center[1] + 40 * np.sin(angle))
            pts.append([x, y])

        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], (0, 255, 255))

        # Convert from BGR to RGB and save
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imsave(output_path, img)

    except Exception as e:
        print(f"Error creating sample image: {e}")


if __name__ == "__main__":
    # Create a sample image if needed
    create_sample_image()

    print("Running simple K-means clustering example...")
    simple_kmeans_example()

    print("\nRunning Iris dataset K-means clustering example...")
    iris_kmeans_example()

    print("\nRunning Olivetti faces K-means clustering example...")
    faces_kmeans_example()

    print("\nRunning image compression example...")
    image_compression_example()

    print("\nRunning color analysis example...")
    color_analysis_example()

    print("\nRunning image segmentation example...")
    segmentation_example()

    print("\nAll examples completed. Results are saved in 'images' and 'analysis' folders.")
