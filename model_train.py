import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


## preparing dataset
df = pd.read_csv("./data/data.csv")
df = df[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement']]

label_encoder = LabelEncoder()

# Apply label encoding to the categorical column
df['city'] = label_encoder.fit_transform(df['city'])
df['statezip'] = label_encoder.fit_transform(df['statezip'])

#splitting the Dataset
X = df.drop(labels='price',axis=1)
Y = df['price']

sc = StandardScaler()
for col in X.columns:
    X[col] = sc.fit_transform(X[col].to_numpy().reshape(-1,1))
    

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


## applying the model
model = LinearRegression()
model.fit(X_train,y_train)

preds = model.predict(X_test)

#Calculating the Loss
from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y_test,preds,squared=False)
MSE = mean_squared_error(y_test,preds)

print("RMSE : ",RMSE)
print("MSE : ",MSE)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)