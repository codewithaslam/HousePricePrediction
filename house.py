import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("HousePricePrediction.csv")

df['SalePrice'] = df['SalePrice'].fillna(
  df['SalePrice'].mean())
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)
df = df.drop("Id", axis=1)

X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

X_train , X_test , y_train , y_test = train_test_split(
    X,y,test_size=0.3,random_state=0
)



import xgboost 
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)

joblib.dump(xgb, "house_price_model.pkl")
print("Model Saved Successfully!")
joblib.dump(X.columns.tolist(), "features.pkl")
print("Features Succesfully Added!!")

y_pred = xgb.predict(X_test)
print("\n After Boosting:----")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAPE: ",mean_absolute_percentage_error(y_test, y_pred))

X_test.to_csv("sample_input.csv", index=False)