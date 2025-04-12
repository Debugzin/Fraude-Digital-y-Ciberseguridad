import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Cargar el archivo Excel
df = pd.read_excel("C:/Users/AGUSTIN LUJAN/Desktop/resultados_modelos.xlsx")  # Ajustá la ruta si es necesario

# Obtener los datos
y_true = df["isFraud"].values
y_pred_auto = df["Autoencoder Prediction"].values
y_pred_if = df["Isolation Forest Prediction"].replace(-1, 0).values
y_pred_svm = df["One-Class SVM Prediction"].replace(-1, 0).values

# Verificar clases
print("🧾 Clases únicas - isFraud:", set(y_true))
print("🧾 Clases únicas - Autoencoder:", set(y_pred_auto))
print("🧾 Clases únicas - Isolation Forest:", set(y_pred_if))
print("🧾 Clases únicas - One-Class SVM:", set(y_pred_svm))

# Diccionario con todos los modelos
modelos = {
    "Autoencoder": y_pred_auto,
    "Isolation Forest": y_pred_if,
    "One-Class SVM": y_pred_svm
}

# 1️⃣ MATRICES DE CONFUSIÓN
plt.figure(figsize=(18, 5))
for i, (nombre, pred) in enumerate(modelos.items()):
    cm = confusion_matrix(y_true, pred)
    plt.subplot(1, 3, i + 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraude", "Fraude"],
                yticklabels=["No Fraude", "Fraude"])
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

# 2️⃣ CURVAS ROC + AUC
plt.figure(figsize=(8, 6))
for nombre, pred in modelos.items():
    fpr, tpr, _ = roc_curve(y_true, pred)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{nombre} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR (Falsos Positivos)")
plt.ylabel("TPR (Verdaderos Positivos)")
plt.title("Curva ROC y AUC")
plt.legend()
plt.grid()
plt.show()

# 3️⃣ BARRAS APILADAS: TP, TN, FP, FN
metricas = {"TP": [], "TN": [], "FP": [], "FN": []}

for nombre, pred in modelos.items():
    cm = confusion_matrix(y_true, pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metricas["TP"].append(tp)
        metricas["TN"].append(tn)
        metricas["FP"].append(fp)
        metricas["FN"].append(fn)
    else:
        print(f"⚠️ Matriz no 2x2 para {nombre}, se omite.")

df_metricas = pd.DataFrame(metricas, index=list(modelos.keys()))

df_metricas.plot(kind='bar', stacked=True, figsize=(10, 6), colormap="viridis")
plt.title("Comparación de TP, TN, FP y FN por modelo")
plt.ylabel("Cantidad")
plt.xlabel("Modelo")
plt.xticks(rotation=0)
plt.legend(title="Métrica")
plt.tight_layout()
plt.show()

# 4️⃣ DIAGRAMA DE DISPERSIÓN: Predicciones vs isFraud
# Reemplazar el gráfico de dispersión por gráfico de barras más útil:
resultados = {
    "Modelo": [],
    "Correctas": [],
    "Incorrectas": []
}

for nombre, pred in modelos.items():
    correctas = (y_true == pred).sum()
    incorrectas = (y_true != pred).sum()
    
    resultados["Modelo"].append(nombre)
    resultados["Correctas"].append(correctas)
    resultados["Incorrectas"].append(incorrectas)

df_resultados = pd.DataFrame(resultados)

df_resultados.set_index("Modelo").plot.bar(
    stacked=True, color=['green', 'red'], figsize=(8, 6)
)

plt.title("Cantidad de predicciones correctas e incorrectas por modelo")
plt.ylabel("Cantidad")
plt.xlabel("Modelo")
plt.xticks(rotation=0)
plt.legend(title="Tipo de predicción")
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
