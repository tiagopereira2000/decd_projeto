import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import re
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB




formandos_terminados = pd.read_excel("Ficheiros2022/Formandos_Terminados2018.xlsx")
# formandos_terminados.head()
print("Ficheiro excel lido")

df = pd.DataFrame(formandos_terminados)
df = df.iloc[:, [0, 4, 6, 7, 8, 14, 19, 21, 22, 24, 25, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]]

df.loc[df['CodHabilitacao'] == 'SL', 'CodHabilitacao'] = '-2'
df.loc[df['CodHabilitacao'] == 'BM', 'CodHabilitacao'] = '20'
df.loc[df['CodHabilitacao'] == 'LC', 'CodHabilitacao'] = '30'
df.loc[df['CodHabilitacao'] == 'MT', 'CodHabilitacao'] = '40'
df['CodHabilitacao'] = df['CodHabilitacao'].str.lstrip('0')
df['CodHabilitacao'] = pd.to_numeric(df['CodHabilitacao'], errors='coerce')
# Preencher NaNs com um valor padrão, como -1
df['CodHabilitacao'] = df['CodHabilitacao'].fillna(-1).astype(int)
df.dropna(inplace=True)

df.loc[df['NivelFormacaoAccao'] == 'L', 'NivelFormacaoAccao'] = '-1'
df.loc[df['NivelFormacaoAccao'] == 'T', 'NivelFormacaoAccao'] = '-2'
df.NivelFormacaoAccao = df.NivelFormacaoAccao.astype(int)

df.Sexo = df.Sexo.astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])

meses = df.iloc[:, -12:]
eficienciaFormacao = meses.sum(axis=1)
df = df.iloc[:, :-12]
df['EficienciaFormacao'] = eficienciaFormacao

df.CodSaidaProfissional = \
    df.CodSaidaProfissional.apply(lambda x: re.sub('\D', '', str(x))).astype(int)

sns.set_theme()
dummies = df.iloc[:, [1, 2, 3, 4, 6, 8, 9, 10, 11]]
sns.heatmap(dummies.corr(),
            xticklabels=dummies.columns,
            yticklabels=dummies.columns)
plt.show()

# idade_categoria = df.iloc[:, [2, 3]].values
# nfa_ch = df.iloc[:, [9, 4]].values
# categoria_csp = df.iloc[:, [3, 10]].values
# ef_categoria = df.iloc[:, [11, 3]].values
# ef_ch = df.iloc[:, [11, 4]].values
# ef_sexo = df.iloc[:, [11, 1]].values

kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(df)

# plt.scatter(idade_categoria[:,1], idade_categoria[:,0], c=kmeans.labels_)
# plt.xlabel("categoria")
# plt.ylabel("idade")
# plt.title("Idade vs Categoria")
# plt.show()
#
# plt.scatter(nfa_ch[:,0], nfa_ch[:,1], c=kmeans.labels_)
# plt.ylabel("Nivel Formacao Accao")
# plt.xlabel("Codigo de Habilitação")
# plt.title("NFA vs CH")
# plt.show()

# plt.scatter(ef_ch[:,1], ef_ch[:,0], c=kmeans.labels_)
# plt.xlabel("Código Habilitação")
# plt.ylabel("Eficiencia de Formação")
# plt.title("CH vs EF")
# plt.show()

# Separar os dados em variáveis descritivas (X) e variável alvo (y)
X = dummies.drop('EficienciaFormacao', axis=1)  # Variáveis descritivas
y = dummies['EficienciaFormacao']  # Variável alvo

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

print("Divisão dos conjuntos de treino e teste")

# lr_clf = LogisticRegression(random_state=0, max_iter=5000)
# scores = cross_val_score(lr_clf, X_train, y_train, cv=5)
# print(scores)
#
#
# tree_clf = tree.DecisionTreeClassifier()
# scores = cross_val_score(tree_clf, X_train, y_train, cv=5)
# print(scores)


# svm_clf = svm.SVC()
# scores = cross_val_score(svm_clf, X_train, y_train, cv=5)
# print(scores)


gnb_clf = GaussianNB()
scores = cross_val_score(gnb_clf, X_train, y_train, cv=5)
print(scores)
