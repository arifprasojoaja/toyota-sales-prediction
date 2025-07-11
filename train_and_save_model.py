import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_csv('penjualan_toyota.csv')

# 2. Buat label kategori
def kategorikan(unit):
    if unit < 20:
        return 'Rendah'
    elif 20 <= unit < 50:
        return 'Sedang'
    else:
        return 'Tinggi'

df['Kategori_Penjualan'] = df['Unit_Terjual'].apply(kategorikan)

# 3. Pilih fitur & label
df = df.drop(columns=['Tanggal', 'Unit_Terjual'])

# 4. One-hot encoding
df_encoded = pd.get_dummies(df, columns=['Model', 'Wilayah'])

# 5. Pisahkan fitur & target
X = df_encoded.drop(columns='Kategori_Penjualan')
y = df_encoded['Kategori_Penjualan']

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Simpan model
with open('toyota_model.sav', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil disimpan sebagai toyota_model.sav")
