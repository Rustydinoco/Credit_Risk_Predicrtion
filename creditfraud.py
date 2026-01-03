
import pandas as pd
import numpy as np

# GENERATE DATASET NASABAH
np.random.seed(101)
n_samples = 5000

data = {
    'umur': np.random.randint(21, 65, n_samples),
    'pendapatan_tahunan': np.random.randint(50_000, 2_000_000, n_samples), 
    'skor_kredit': np.random.randint(300, 850, n_samples),
    'jumlah_pinjaman': np.random.randint(10_000, 500_000, n_samples),
}

df = pd.DataFrame(data)


probabilitas_default = (
    (850 - df['skor_kredit']) / 850 * 0.4 +
    (df['jumlah_pinjaman'] / df['pendapatan_tahunan']) * 0.4 +
    (df['umur'] > 40).astype(int) * 0.05
)
# Tambah noise acak biar gak gampang ditebak 100%
probabilitas_default += np.random.normal(0, 0.05, n_samples)
df['status_default'] = (probabilitas_default > 0.45).astype(int)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('status_default', axis=1)
y = df['status_default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train_scaled, y_train)

import joblib

joblib.dump(model, 'model_credit3.pkl')
joblib.dump(scaler, 'scaler_credit3.pkl')