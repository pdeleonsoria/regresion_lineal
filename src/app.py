from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
#CARGAR DATOS Y EDA 

#Importar el csv:

url= "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"

df= pd.read_csv(url)
#Creo 4 columnas, una para cada región y elimino la columna región 
df_dummies = pd.get_dummies(df['region'], prefix='region')
df = pd.concat([df, df_dummies], axis=1)
df.head()
df = df.drop('region', axis=1)

#Eliminar duplicados:
df.drop_duplicates()
df.head()

#Convierto los valores cualitativos en cuantitativos:

le= LabelEncoder()

df["sex"]= le.fit_transform(df["sex"])
df["smoker"]= le.fit_transform(df["smoker"])
df["region_northeast"]= le.fit_transform(df["region_northeast"])
df["region_northwest"]= le.fit_transform(df["region_northwest"])
df["region_southeast"]= le.fit_transform(df["region_southeast"])
df["region_southwest"]= le.fit_transform(df["region_southwest"])
#

variables=["age","sex",	"bmi",	"children",	"smoker",	"charges",	"region_northeast",	"region_northwest",	"region_southeast",	"region_southwest"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(df[variables])
df_escalado= pd.DataFrame(scal_features, index = df.index, columns = variables)
df_escalado.head()

#TRAIN Y TEST

X = df_escalado.drop("charges", axis=1) 
y = df_escalado["charges"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selection_model = SelectKBest(score_func=mutual_info_regression, k=5)
selection_model.fit(X_train, y_train)
selec = selection_model.get_support()

X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns=X_train.columns.values[selec])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns=X_test.columns.values[selec])
X_train_sel.head()

#Uso f-ANOVA porque la variable objeto es continua y no funciona asi que cambio a mutual_info_regression

X_test_sel.head()


# Visualización de regresión
fig, axis = plt.subplots(2, 2, figsize=(14, 12))

sns.regplot(x=df_escalado['age'], y=df_escalado['charges'], ax=axis[0, 0])
axis[0, 0].set_title('Regresión: Edad vs. Cargos')

sns.regplot(x=df_escalado['bmi'], y=df_escalado['charges'], ax=axis[0, 1])
axis[0, 1].set_title('Regresión: BMI vs. Cargos')

sns.regplot(x=df_escalado['children'], y=df_escalado['charges'], ax=axis[1, 0])
axis[1, 0].set_title('Regresión: Hijos vs. Cargos')

sns.regplot(x=df_escalado['smoker'], y=df_escalado['charges'], ax=axis[1, 1])
axis[1, 1].set_title('Regresión: Fumador vs. Cargos')

plt.tight_layout()
plt.show()

model = LinearRegression()
model.fit(X_train_sel, y_train)

y_pred = model.predict(X_test_sel)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.6f}")
print(f"R2: {r2:.4f}")

#Los valores de R2 y de MSE son buenos y no hace falta optimizar el modelo


#Resumen de correlación de datos
X = df.drop(["charges", "region_northeast"], axis=1)
y = df["charges"]
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

df.head()
