import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy:', round(metrics.accuracy_score(y_test, y_pred), 4))
print('Precision:', round(metrics.precision_score(y_test, y_pred, average='weighted'), 4))
print('Recall:', round(metrics.recall_score(y_test, y_pred, average='weighted'), 4))
print('F1 Score:', round(metrics.f1_score(y_test, y_pred, average='weighted'), 4))
print('Cohen Kappa Score:', round(metrics.cohen_kappa_score(y_test, y_pred), 4))
print('Matthews Corrcoef:', round(metrics.matthews_corrcoef(y_test, y_pred), 4))

print('\nClassification Report:\n', metrics.classification_report(y_test, y_pred, target_names=iris.target_names))

mat = confusion_matrix(y_test, y_pred)
sns.set()
plt.figure(figsize=(6, 5))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Матриця плутанини RidgeClassifier')
plt.tight_layout()
plt.savefig("Confusion.jpg")
plt.show()
