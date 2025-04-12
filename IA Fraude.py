import os
import time

# Desactivar las optimizaciones de oneDNN antes de importar TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Usar solo CPU

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

# Configurar la CPU para usar múltiples hilos
physical_devices = tf.config.list_physical_devices('CPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices, 'CPU')
    tf.config.threading.set_intra_op_parallelism_threads(12)
    tf.config.threading.set_inter_op_parallelism_threads(12)

print("Configuración de CPU completada.")

# Cargar los datos
print("Cargando los archivos CSV...")
df_transaction = pd.read_csv(r'C:\Users\AGUSTIN LUJAN\Downloads\ieee-fraud-detection\train_transaction.csv')
df_identity = pd.read_csv(r'C:\Users\AGUSTIN LUJAN\Downloads\ieee-fraud-detection\train_identity.csv')
print(f"Datos cargados: {len(df_transaction)} transacciones, {len(df_identity)} identidades.")

# Fusionar los datos
print("Fusionando los dataframes...")
df = pd.merge(df_transaction, df_identity, on='TransactionID', how='left')
print(f"Fusionado completado. Tamaño final: {df.shape}")

# Preprocesamiento
print("Preprocesando los datos...")
features = df.drop(columns=['isFraud', 'TransactionID'], errors='ignore')  # Ignorar error si faltan columnas
labels = df.get('isFraud', pd.Series([0] * len(df)))  # Manejar si falta 'isFraud'

# Convertir todo a numérico
features_numeric = features.select_dtypes(include=[np.number]).fillna(0)

# Normalización
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_numeric)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# ===========================
# 🔹 MODELO AUTOENCODER 🔹
# ===========================
print("Entrenando Autoencoder...")
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar con medición de tiempo
start_time = time.time()
autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, validation_data=(X_test, X_test), verbose=1)
end_time = time.time()
print(f"Autoencoder entrenado en {end_time - start_time:.2f} segundos.")

# ===========================
# 🔹 MODELO ISOLATION FOREST 🔹
# ===========================
print("🌲 Entrenando Isolation Forest...")
start_time = time.time()
iso_forest = IsolationForest(contamination=0.1, n_jobs=-1)
iso_forest.fit(X_train)
end_time = time.time()
print(f"Isolation Forest entrenado en {end_time - start_time:.2f} segundos.")

# ===========================
# 🔹 MODELO ONE-CLASS SVM 🔹
# ===========================
print("Entrenando One-Class SVM...")

# Tomar una muestra de 50,000 filas para evitar bloqueos
sample_size = min(50000, len(X_train))
X_train_sample = X_train[:sample_size]

start_time = time.time()
one_class_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale", verbose=True)
one_class_svm.fit(X_train_sample)
end_time = time.time()

print(f"One-Class SVM entrenado en {end_time - start_time:.2f} segundos.")

# ===========================
# 🔹 PREDICCIONES Y GUARDADO 🔹
# ===========================
print("Realizando predicciones...")

# Asegurar que X_test es un DataFrame con índices originales
X_test_df = pd.DataFrame(X_test, index=df_transaction.index[-len(X_test):])

# Predicciones con Autoencoder
reconstructed = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)
threshold = np.percentile(mse, 95)
y_pred_autoencoder = (mse > threshold).astype(int)

# Predicciones con Isolation Forest
y_pred_iso = iso_forest.predict(X_test)

# Predicciones con One-Class SVM
y_pred_svm = one_class_svm.predict(X_test)

# Crear un DataFrame con los resultados asegurando índices correctos
results = pd.DataFrame({
    'TransactionID': df_transaction['TransactionID'].iloc[X_test_df.index],
    'isFraud': y_test.values,  # Asegurar alineación
    'Autoencoder Prediction': y_pred_autoencoder,
    'Isolation Forest Prediction': y_pred_iso,
    'One-Class SVM Prediction': y_pred_svm
})

# Ruta donde se guardará el archivo
output_path = r'C:\Users\AGUSTIN LUJAN\Desktop\resultados_modelos.xlsx'

# Guardar el archivo Excel
results.to_excel(output_path, index=False)

print(f"Resultados guardados en '{output_path}'.")
