# === 1. Import des bibliothèques ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# === 2. Chargement des données ===
df = pd.read_csv("diabetes.csv")
print("Aperçu des données :")
print(df.head())

# === 3. Nettoyage & Prétraitement ===
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, pd.NA)
print("\nPourcentage de valeurs manquantes :")
print(df.isna().mean() * 100)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())

# === 4. EDA & Déséquilibre ===
plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=df, palette='Set2')
plt.title("Répartition des cas (0 = non diabétique, 1 = diabétique)")
plt.tight_layout()
plt.show()

print("\nStatistiques descriptives :")
print(df.groupby('Outcome').describe())

# === 5. Feature Engineering (Standardisation) ===
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# === 6. Split Train / Validation / Test ===
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled_df, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("\nTailles des datasets :")
print(f"Train      : {X_train.shape[0]}")
print(f"Validation : {X_val.shape[0]}")
print(f"Test       : {X_test.shape[0]}")

# === 7. Gestion du déséquilibre (option: class_weight) ===
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# === 8. Évaluation sur le set de validation ===
y_val_pred = model.predict(X_val)
print("\n📊 Évaluation sur le jeu de validation :")
print(f"Accuracy  : {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision : {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall    : {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 Score  : {f1_score(y_val, y_val_pred):.4f}")
print("\nRapport de classification (Validation) :")
print(classification_report(y_val, y_val_pred, digits=4))

# === 9. Évaluation finale sur le test set ===
y_test_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Matrice de confusion - Test Set")
plt.tight_layout()
plt.show()
print("\nRapport de classification (Test) :")
print(classification_report(y_test, y_test_pred, digits=4))

# === 10. Explainability : Importance des features ===
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0]
})
coef_df['abs'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='abs', ascending=False)

print("\n📌 Importance des variables (coefs) :")
print(coef_df[['Feature', 'Coefficient']])

plt.figure(figsize=(8,6))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='viridis')
plt.title("Importance des variables - Régression Logistique")
plt.tight_layout()
plt.show()

# === Analyse des erreurs ===
fp_index = (y_test == 0) & (y_test_pred == 1)
fn_index = (y_test == 1) & (y_test_pred == 0)

print(f"\n❌ Faux positifs : {fp_index.sum()} | ❌ Faux négatifs : {fn_index.sum()}")
print("\nExemples de faux positifs :")
print(X_test[fp_index].head())

print("\nExemples de faux négatifs :")
print(X_test[fn_index].head())
