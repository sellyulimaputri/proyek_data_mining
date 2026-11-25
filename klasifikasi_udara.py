import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. LOAD DATASET
NAMA_FILE_CSV = 'ispu_dki_all.csv' # Ganti nama file di bawah ini sesuai dengan nama file CSV yang ada di foldermu

print("1. Membaca dataset...")
df = pd.read_csv(NAMA_FILE_CSV)
print(f"   Ukuran dataset awal: {df.shape[0]} baris, {df.shape[1]} kolom\n")

# 2. DATA CLEANING & FEATURE SELECTION
print("2. Melakukan Data Cleaning & Feature Selection...")
kolom_hapus = ['tanggal', 'stasiun', 'max', 'critical', 'pm25'] # Hapus kolom metadata & pm25 (karena data awal tahun banyak yang kosong)
df_clean = df.drop(columns=[col for col in kolom_hapus if col in df.columns])

df_clean.replace(['---', 'TIDAK ADA DATA', '-'], np.nan, inplace=True) # Ubah tanda '---' atau teks tidak valid menjadi NaN

df_clean = df_clean.drop_duplicates() # Hapus baris duplikat dan baris yang kategori targetnya kosong
df_clean = df_clean.dropna(subset=['categori'])

fitur = ['pm10', 'so2', 'co', 'o3', 'no2'] # Ubah kolom parameter polutan menjadi tipe numerik (float)
for col in fitur:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

jumlah_per_kelas = df_clean['categori'].value_counts() # Hapus kategori langka yang jumlah datanya terlalu sedikit (< 6 baris) agar SMOTE tidak error
kelas_valid = jumlah_per_kelas[jumlah_per_kelas >= 6].index
df_clean = df_clean[df_clean['categori'].isin(kelas_valid)]

print("   Jumlah missing value per fitur sebelum imputasi:")
print(df_clean[fitur].isnull().sum().to_string())
print()

# 3. MISSING VALUE HANDLING & ENCODING
print("3. Menangani Missing Value & Melakukan Label Encoding...")
imputer = SimpleImputer(strategy='median') # Isi nilai kosong dengan nilai tengah (Median)
df_clean[fitur] = imputer.fit_transform(df_clean[fitur])

df_clean.to_csv('ispu_dki_bersih.csv', index=False) # Simpan dataset yang sudah bersih ke file CSV baru
print("   Dataset bersih berhasil disimpan sebagai 'ispu_dki_bersih.csv'!\n")

X = df_clean[fitur]
y = df_clean['categori']

le = LabelEncoder() # Ubah label teks kategori menjadi angka
y_encoded = le.fit_transform(y)

print("   Mapping Label Kategori:")
for i, kelas in enumerate(le.classes_):
    print(f"   - {i} : {kelas}")
print()

# 4. DATA SPLITTING, FEATURE SCALING & SMOTE
print("4. Membagi Data (80:20), Scaling, dan SMOTE...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler() # Feature Scaling (Wajib untuk SVM)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42) # Data Balancing dengan SMOTE (Hanya pada data training)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print(f"   Data Training sebelum SMOTE : {X_train.shape[0]} baris")
print(f"   Data Training sesudah SMOTE : {X_train_bal.shape[0]} baris")
print(f"   Data Testing                : {X_test.shape[0]} baris\n")

# 5. TRAINING MODEL & 5-FOLD CROSS VALIDATION
print("5. Melatih Model Random Forest & SVM...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='rbf', random_state=42)

rf_cv = cross_val_score(rf_model, X_train_bal, y_train_bal, cv=5, scoring='accuracy') # Validasi Silang (5-Fold Cross Validation)
svm_cv = cross_val_score(svm_model, X_train_bal, y_train_bal, cv=5, scoring='accuracy')

print(f"   Rata-rata Akurasi CV Random Forest : {rf_cv.mean()*100:.2f}%")
print(f"   Rata-rata Akurasi CV SVM           : {svm_cv.mean()*100:.2f}%\n")

rf_model.fit(X_train_bal, y_train_bal) # Latih model final dengan seluruh data training
svm_model.fit(X_train_bal, y_train_bal)

# 6. EVALUASI MODEL & SIMPAN GRAFIK CONFUSION MATRIX
print("6. Evaluasi pada Data Testing...\n")
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_svm = svm_model.predict(X_test_scaled)

print("HASIL EVALUASI: RANDOM FOREST")
print(f"Akurasi Testing: {accuracy_score(y_test, y_pred_rf)*100:.2f}%\n")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

print("HASIL EVALUASI: SUPPORT VECTOR MACHINE (SVM)")
print(f"Akurasi Testing: {accuracy_score(y_test, y_pred_svm)*100:.2f}%\n")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5)) # Membuat dan menyimpan grafik Confusion Matrix otomatis ke folder proyek

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title('Confusion Matrix - Random Forest')
axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Aktual')

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_title('Confusion Matrix - SVM')
axes[1].set_xlabel('Prediksi')
axes[1].set_ylabel('Aktual')

plt.tight_layout()
plt.savefig('hasil_confusion_matrix.png', dpi=300)
print("Grafik berhasil disimpan sebagai 'hasil_confusion_matrix.png' di foldermu!")
plt.show()