# Import necessary packages
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

# We assign the image paths
dataset_path = r"C:\Users\medha\Downloads\traffic_signs_dataset"  # Update with your dataset path
templates_path = r"C:\Users\medha\Downloads\final_templates_project"  # Final combined templates folder
output_folder = r"C:\Users\medha\Downloads\final_predicted_images"
os.makedirs(output_folder, exist_ok=True)

# The classes we want to predict
classes = ["noparking", "railroad", "speedlimit", "stopsign", "yield"]

# We load the templates from the folder
def load_templates(templates_path, classes):
    templates = {}
    for cls in classes:
        cls_path = os.path.join(templates_path, cls)
        if os.path.exists(cls_path):
            templates[cls] = [
                cv2.imread(os.path.join(cls_path, file), 0)
                for file in os.listdir(cls_path)
                if file.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
    return templates

templates = load_templates(templates_path, classes)

# Function to normalize images
def normalize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# Function for template matching
def template_matching(image, templates):
    gray_image = normalize_image(image)
    best_match = ("unknown", 0)

    for cls, cls_templates in templates.items():
        for template in cls_templates:
            if template is None:
                continue

            # Resizing templates
            if template.shape[0] > gray_image.shape[0] or template.shape[1] > gray_image.shape[1]:
                scale_factor = min(
                    gray_image.shape[0] / template.shape[0],
                    gray_image.shape[1] / template.shape[1],
                )
                new_size = (int(template.shape[1] * scale_factor), int(template.shape[0] * scale_factor))
                template = cv2.resize(template, new_size, interpolation=cv2.INTER_AREA)

            res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > best_match[1]:
                best_match = (cls, max_val)

    return best_match

# Function for geometry and color detection
def geometric_color_detection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red color range for stop signs
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Detect stop signs (octagons)
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 8:
            return "stopsign"

    return None

# Function for Ensemble detection
def ensemble_detection(image, templates):
    tm_label, tm_confidence = template_matching(image, templates)
    gd_label = geometric_color_detection(image)

    # Ensemble logic
    if gd_label:
        return gd_label  # Geometric/color detection takes precedence
    else:
        return tm_label  # Fall back to template matching

# Function for evaluation of classification
def evaluate_dataset():
    true_labels = []
    predicted_labels = []
    misclassified_paths = [] 
    f = 0 

    for label in classes:
        folder_path = os.path.join(dataset_path, label)
        if not os.path.exists(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)
            if image is None:
                continue
            if f == 0:
                start_time = time.time()
                predicted_label = ensemble_detection(image, templates)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Time taken: " ,elapsed_time)
                f = 1

            predicted_label = ensemble_detection(image, templates)
            true_labels.append(label)
            predicted_labels.append(predicted_label)

            # Save misclassified image paths
            if predicted_label != label:
                misclassified_paths.append(file_path)

            # Save predicted image
            output_file = os.path.join(output_folder, f"{label}_as_{predicted_label}_{file_name}")
            cv2.putText(image, f"Pred: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(output_file, image)

    return true_labels, predicted_labels, misclassified_paths

true_labels, predicted_labels, misclassified_paths = evaluate_dataset()

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=classes))

conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# We plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# We log misclassified paths
print("\nMisclassified Image Paths:")
for path in misclassified_paths:
    print(path)
