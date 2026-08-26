import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#  Dataset (troque por pd.read_csv("transacoes.csv")
rng = np.random.default_rng(42)
n = 5000
df = pd.DataFrame({
    "valor": rng.gamma(2, 150, n),
    "hora": rng.integers(0, 24, n),
    "distancia_km": np.abs(rng.normal(10, 30, n)),
})
df["fraude"] = ((df["distancia_km"] > 60) & (df["hora"] < 6)).astype(int)
df.loc[rng.random(n) < 0.02, "fraude"] = 1 

print("Distribuicao_original:\n", df["fraude"].value_counts())

#  EDA
sns.countplot(x="fraude", data=df)
plt.title("Transacoes: Legitima (0) x Fraude (1)")
plt.savefig("/home/claude/distribuicao.png")
plt.close()

# Balanceamento por undersampling (iguala as classes)
legitimas = df[df["fraude"] == 0]
fraudes = df[df["fraude"] == 1]
legitimas_bal = legitimas.sample(len(fraudes), random_state=42)
df_bal = pd.concat([legitimas_bal, fraudes]).sample(frac=1, random_state=42)

print("\nDistribuicao balanceada:\n", df_bal["fraude"].value_counts())

#  Treino e teste

X = df_bal.drop(columns="fraude")
y = df_bal["fraude"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

#  Avaliacao

print("\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusao")
plt.savefig("/home/claude/matriz_confusao.png")
plt.close()

