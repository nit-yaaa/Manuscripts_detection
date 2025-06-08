import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Function to load images from directories
def load_images_from_folders(folders):
    images = []
    labels = []
    for label, folder in enumerate(folders.keys()):
        print(f"Loading images for {folder} from {folders[folder]}...")
        for filename in os.listdir(folders[folder]):
            img_path = os.path.join(folders[folder], filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

# Function to extract SIFT descriptors from images
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors.astype(np.float32))  # Convert to float32 to avoid type mismatch
    return descriptors_list

# Function to compute histograms using the BoVW model
def compute_histograms(images, bovw_model, num_clusters):
    sift = cv2.SIFT_create()
    histograms = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors = descriptors.astype(np.float32)  # Convert to float32
            visual_words = bovw_model.predict(descriptors)
            histogram, _ = np.histogram(visual_words, bins=np.arange(num_clusters + 1), density=True)
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(num_clusters))  # If no descriptors, add an empty histogram
    return np.array(histograms)

# Define paths for different manuscript datasets
# add your dataset paths here
folders = {
#     'Bengali': r"D:\Manuscript_Detection\train_data\augment_bengali",
#     'Kannada': r"D:\Manuscript_Detection\train_data\augment_nitya_kannada",
#     'Telugu': r"D:\Manuscript_Detection\train_data\augment_karthik_telugu",
#     'Tamil': r"D:\Manuscript_Detection\train_data\augmented_karthik_tamil"
}

# Load images and corresponding labels
images, labels = load_images_from_folders(folders)
print(f"Total images loaded: {len(images)}")

# Extract SIFT descriptors
print("Extracting SIFT features...")
descriptors_list = extract_sift_features(images)
descriptors = np.vstack(descriptors_list)  # Stack all descriptors into one numpy array
print(f"Total descriptors extracted: {descriptors.shape[0]}")

# BoVW model parameters
num_clusters = 50
print(f"Building BoVW model with {num_clusters} clusters...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
bovw_model = kmeans.fit(descriptors)
print(f"BoVW model built with {num_clusters} clusters.")

# Compute histograms for each image using the BoVW model
print("Computing histograms for each image...")
histograms = compute_histograms(images, bovw_model, num_clusters)
print(f"Histograms computed for {len(histograms)} images.")

# Convert labels to numpy array for classification
labels = np.array(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(histograms, labels, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Train a Support Vector Machine (SVM) classifier
print("Training SVM classifier...")
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predict on the test set
print("Predicting on test set...")
y_pred = classifier.predict(X_test)

# Calculate overall evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print overall evaluation results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generate language-wise classification report
class_labels = ['Bengali', 'Kannada', 'Telugu', 'Tamil']
report = classification_report(y_test, y_pred, target_names=class_labels)
print("\nClassification Report (Language-wise):")
print(report)
