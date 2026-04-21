import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


data = []
labels = []

# =========================
# LOAD DATASET
# =========================
dataset_path = "dataset"

for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize & grayscale
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Flatten
        data.append(img.flatten())
        labels.append(label)

# =========================
# PREPROCESSING
# =========================
data = np.array(data)
labels = np.array(labels)

# Normalisasi (biar akurat)
data = data / 255.0

# Info dataset
print("Jumlah data:", len(data))
print("Jumlah label:", len(labels))

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL KNN
# =========================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# =========================
# HITUNG AKURASI
# =========================
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Akurasi:", accuracy * 100, "%")

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("Confusion Matrix:")
print(cm)

# =========================
# TEST GAMBAR BARU
# =========================
test_img = cv2.imread("test/test.jpg")

if test_img is None:
    print("Gambar test tidak ditemukan!")
    exit()

test_img = cv2.resize(test_img, (64, 64))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# Flatten + reshape + normalisasi
test_data = test_img.flatten().reshape(1, -1) / 255.0

prediction = knn.predict(test_data)

plt.imshow(test_img, cmap='gray')
plt.title(f"Hasil: {prediction[0]}")
plt.axis('off')
plt.show()

print("Hasil:", prediction[0])