# Import library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Membaca Data
file_path = 'DataTasya.xlsx'
data = pd.read_excel(file_path)

# Membersihkan data
data.columns = data.iloc[0]  # Mengatur ulang kolom
data = data.drop(0).reset_index(drop=True)  # Menghapus baris pertama
data.columns = ['No', 'Provinsi', 'Jumlah Balita (Anak)', 'Pendek (Anak)', 'Sangat Pendek (Anak)', 'Persentase Kasus Stunting (%)']
data = data.drop(columns=['No', 'Provinsi']).dropna()  # Menghapus kolom tidak perlu dan baris kosong
data = data.astype(float)  # Mengonversi ke tipe numerik

# Menampilkan data untuk memastikan
print(data.head())

# 2. Normalisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. Implementasi DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan_labels = dbscan.fit_predict(data_scaled)

# Menambahkan hasil kluster DBSCAN ke dataframe
data['DBSCAN Cluster'] = dbscan_labels
print("DBSCAN Clusters:\n", data['DBSCAN Cluster'].value_counts())

# Plot hasil DBSCAN
plt.figure(figsize=(10, 6))
plt.scatter(data['Jumlah Balita (Anak)'], data['Pendek (Anak)'], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Jumlah Balita (Anak)')
plt.ylabel('Pendek (Anak)')
plt.colorbar(label='Cluster')
plt.show()

# 4. Implementasi Fuzzy C-Means
# Transpose data karena skfuzzy c-means bekerja dengan data berbentuk (features, samples)
data_transposed = np.transpose(data_scaled)
n_clusters = 3  # Anda bisa menyesuaikan jumlah kluster

# Menjalankan algoritma Fuzzy C-Means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_transposed, n_clusters, 2, error=0.005, maxiter=1000, init=None)

# Menentukan hasil klustering untuk tiap data
fcm_labels = np.argmax(u, axis=0)
data['Fuzzy C-Means Cluster'] = fcm_labels

print("Fuzzy C-Means Clusters:\n", data['Fuzzy C-Means Cluster'].value_counts())

# Plot hasil Fuzzy C-Means
plt.figure(figsize=(10, 6))
plt.scatter(data['Jumlah Balita (Anak)'], data['Pendek (Anak)'], c=fcm_labels, cmap='viridis')
plt.title('Fuzzy C-Means Clustering')
plt.xlabel('Jumlah Balita (Anak)')
plt.ylabel('Pendek (Anak)')
plt.colorbar(label='Cluster')
plt.show()
