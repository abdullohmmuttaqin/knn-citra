import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime


# Inisialisasi data
# =================

data = []
labels = []

dataset_path = "dataset"

# Load dataset
# =================
for label in os.listdir(dataset_path):
    #label = nama folder (daunSehat / daunSakit)

    folder_path = os.path.join(dataset_path, label)
    #path ke folder

    for img_name in os.listdir(folder_path):
        #nama file gambar

        img_path = os.path.join(folder_path, img_name)
        #path lengkap ke gambar

        img = cv2.imread(img_path)
        #baca gambar

        if img is None:
            #jikalau gambar error
            continue

        #resize gambar jadi 64x64
        img = cv2.resize(img, (64, 64))

        #ubah ke grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #ubah jadi array 1 dimensi
        data.append(img.flatten())

        #simpan label
        labels.append(label)

# Konversi data
# =================

data = np.array(data)
labels = np.array(labels)

print("Jumlah data:", len(data))
print("Jumlah label", len(labels))

# Split data
# =================

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Model KNN
# =================

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)

# Model SVM
# =================

svm = SVC(kernel='linear')
svm.fit(x_train, y_train)

y_pred_svm = svm.predict(x_test)

# Akurasi
# =================

acc_knn = accuracy_score(y_test, y_pred_knn)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("Akurasi KNN:", acc_knn * 100, "%")
print("Akurasi SVM:", acc_svm * 100, "%")

# Confusion matrix
# =================

cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_svm = confusion_matrix(y_test, y_pred_svm)

# Visualisasi
# =================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title("KNN")

plt.subplot(1, 2, 2)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds')
plt.title("SVM")

plt.tight_layout()

now = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"output/confusion_matrix_{now}.png")

plt.show()

# Test gambar baru
# =================

test_img = cv2.imread("test/test.jpg")

test_img = cv2.resize(test_img, (64, 64))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

test_data = test_img.flatten().reshape(1, -1)

pred_knn = knn.predict(test_data)
pred_svm = svm.predict(test_data)

print("Prediksi KNN:", pred_knn[0])
print("Prediksi SVM:", pred_svm[0])

# Tampilkan gambar test
# =================

plt.imshow(test_img, cmap='gray')
plt.title(f"KNN: {pred_knn[0]} | SVM: {pred_svm[0]}")
plt.axis('off')
plt.show()