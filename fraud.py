import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from mpl_toolkits.mplot3d import Axes3D
import time

tic = time.time()
full_data = pd.read_csv("creditcard.csv")
# full_data = full_data.sample(frac = 1)       	randomize the whole dataset
full_features = full_data.drop(["Time","Class"], axis = 1)

full_labels = pd.DataFrame(full_data[["Class"]])

full_features_array = full_features.values
full_labels_array = full_labels.values

# Normalizing Data
train_features, test_features, train_labels, test_labels = train_test_split(full_features_array, full_labels_array, train_size = 0.90)
train_features = normalize(train_features)
test_features = normalize(test_features)

# k_means_classification --> k_means_clustering, confsion_matrix, reassigning
kmeans = KMeans(n_clusters = 2, random_state = 0, algorithm = "elkan", max_iter = 10000, n_jobs = -1)
kmeans.fit(train_features)
kmeans_predicted_train_labels = kmeans.predict(train_features)

# Confusion Matrix
# tn fp
# fn tp
print("tn --> true negatives")
print("fp --> false positives")
print("fn --> false negatives")
print("tp --> true positives")

tn, fp, fn, tp = confusion_matrix(train_labels, kmeans_predicted_train_labels).ravel()
reassignflag = False

if tn + tp < fn + fp:
	# Clustering is opposite of Original Classification
	reassignflag = True
kmeans_predicted_test_labels = kmeans.predict(test_features)
if reassignflag:
	kmeans_predicted_test_labels = 1 - kmeans_predicted_test_labels

# Calculating Confusion Matrix for kmeans
tn, fp, fn, tp = confusion_matrix(test_labels, kmeans_predicted_test_labels).ravel()

# Scoring kmeans
kmeans_accuracy_score = accuracy_score(test_labels,kmeans_predicted_test_labels)
kmeans_precison_score = precision_score(test_labels,kmeans_predicted_test_labels)
kmeans_recall_score = recall_score(test_labels,kmeans_predicted_test_labels)
kmeans_f1_score = f1_score(test_labels,kmeans_predicted_test_labels)

# Printing
print("\nK-Means")
print("Confusion Matrix")
print("tn =", tn, "fp =",fp)
print("fn =", fn, "tp =",tp)
print("Scores")
print("Accuracy -->", kmeans_accuracy_score)
print("Precison -->", kmeans_precison_score)
print("Recall -->", kmeans_recall_score)
print("F1 -->", kmeans_f1_score)

# k_nearest_neighbours_classification:
knn = KNeighborsClassifier(n_neighbors = 5, algorithm = "kd_tree", n_jobs = -1)
knn.fit(train_features, train_labels.ravel())
knn_predicted_test_labels = knn.predict(test_features)

# Calculating Confusion Matrix for knn
tn, fp, fn, tp = confusion_matrix(test_labels, knn_predicted_test_labels).ravel()

# Scoring knn
knn_accuracy_score = accuracy_score(test_labels, knn_predicted_test_labels)
knn_precison_score = precision_score(test_labels, knn_predicted_test_labels)
knn_recall_score = recall_score(test_labels, knn_predicted_test_labels)
knn_f1_score = f1_score(test_labels, knn_predicted_test_labels)

# Printing
print("\nK-Nearest Neighbours")
print("Confusion Matrix")
print("tn =", tn, "fp =", fp)
print("fn =", fn, "tp =", tp)
print("Scores")
print("Accuracy -->", knn_accuracy_score)
print("Precison -->", knn_precison_score)
print("Recall -->", knn_recall_score)
print("F1 -->", knn_f1_score)

# ROC Curve for Knn
fpr, tpr, threshold = roc_curve(test_labels, knn_predicted_test_labels)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

# Time Elapsed
toc = time.time()
elapsedtime = toc - tic
print("\nTime Taken : "+str(elapsedtime)+"seconds")
