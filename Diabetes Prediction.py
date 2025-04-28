# IMPORT LIBRARY
import pandas as pd
import numpy as np

# IMPORT CSV FILE AS A DATAFRAME
df = pd.read_csv(r'https://github.com/YBI-Foundation/Dataset/raw/main/Diabetes.csv')

#GET FIRST ROWS
df.head()
df.info()
df.describe()
df.columns
df.shape

# GET UNIQUE VALUES IN Y VARIABLE
df['diabetes'].value_counts()
df.groupby('diabetes').mean()

# Define target variable(y) and features(X)
y = df['diabetes']
y.shape
y
X = df[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi','dpf', 'age']]
X.shape
X

# Get X variable Standardised
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X = mm.fit_transform(X)
X

# Get train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# Get Model Train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

# GET MODEL PREDICTION
y_pred = model.predict(X_test)
y_pred.shape
y_pred

# Get probability of each predicted class
model.predict_proba(X_test)

# Get Model Evaluation
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

# Get Future Prediction
X_new = df.sample(1)
X_new
X_new.shape
X_new = X_new.drop('diabetes',axis=1)
X_new
X_new = mm.fit_transform(X_new)
y_pred_new = model.predict(X_new)
y_pred_new
model.predict_proba(X_new)
