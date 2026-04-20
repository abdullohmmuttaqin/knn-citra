import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

data = []
labels = []

# Load dataset
dataset_path = "dataset"

for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data.append(img.flatten())
        labels.append(label)

# Ubah ke array
data = np.array(data)

# Model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data, labels)

# =========================
# TEST GAMBAR
# =========================
test_img = cv2.imread("test/test.jpg")

if test_img is None:
    print("Gambar test tidak ditemukan!")
    exit()

test_img = cv2.resize(test_img, (64, 64))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# FIX DI SINI 🔥
test_data = test_img.flatten().reshape(1, -1)

prediction = knn.predict(test_data)

print("Hasil:", prediction[0])