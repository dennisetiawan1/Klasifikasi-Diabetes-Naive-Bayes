import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv('data/diabetes (2).csv')

# Pisahkan fitur dan label
X = df.drop(columns='Outcome')
y = df['Outcome']

# Tangani data tidak seimbang
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Latih model
model = GaussianNB()
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, 'model/naive_bayes_model.pkl')
print("Model berhasil disimpan sebagai 'naive_bayes_model.pkl'")
