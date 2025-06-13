import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt

# 1) Leitura como CSV (mesmo nome .xls)
df = pd.read_csv('healthcare-dataset-stroke-data.xls')

# 2) Limpeza de colunas
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# 3) Tratamento de categóricas
cat_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 4) Imputação de numéricas
num_cols = ['age','hypertension','heart_disease','avg_glucose_level','bmi']
imp = SimpleImputer(strategy='median')
df[num_cols] = imp.fit_transform(df[num_cols])

# 5) Separar X e y
target = 'stroke'
X = df.drop(target, axis=1)
y = df[target]

# 6) Balanceamento SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# 7) Padronização
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# 8) Divisão treino/teste
X_tr, X_te, y_tr, y_te = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 9) Treinamento
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
rf.fit(X_tr, y_tr)
lr.fit(X_tr, y_tr)

# 10) Avaliação
print("=== Random Forest ===")
print(classification_report(y_te, rf.predict(X_te)))
print("=== Logistic Regression ===")
print(classification_report(y_te, lr.predict(X_te)))

# 11) Importância de variáveis (RF)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns.tolist()
ordered = [features[i] for i in indices]
print(features)
plt.figure()
plt.barh(ordered, importances[indices])
plt.title("Importância das Variáveis (Random Forest)")
plt.xlabel("Importância")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

feature_names = X.columns.tolist()
# --- Gráfico Regressão Logística ---
coefs = lr.coef_[0]                                   # coeficientes por feature
indices_lr = np.argsort(np.abs(coefs))[::-1]          # ordena por magnitude
ordered_lr = [feature_names[i] for i in indices_lr]   
values_lr  = coefs[indices_lr]

plt.figure()
plt.barh(ordered_lr, values_lr)
plt.title("Coeficientes do Modelo (Logistic Regression)")
plt.xlabel("Valor do Coeficiente")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 12) Salvar modelos
joblib.dump(rf, 'rf_avc_model.pkl')
joblib.dump(lr, 'lr_avc_model.pkl')
print("Modelos salvos em 'rf_avc_model.pkl' e 'lr_avc_model.pkl'")
