import csv
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

model = Perceptron()

# buscar el archivo de notas
with open("valores.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# Dividir entre entrenamiento y valores reales
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]

X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)


model.fit(X_training, y_training)

predictions = model.predict(X_testing)

#guardar valores
correct = (y_testing == predictions).sum()
incorrect = (y_testing != predictions).sum()
total = len(predictions)

#mostrar en pantalla

print("Resultados de: Perceptron")
print(f"Correctos: {correct}")
print(f"Incorrecto: {incorrect}")
print(f"Precision: {100 * correct / total:.2f}%")
