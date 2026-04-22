import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


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
