# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NAGALAPURAM HASIF
RegisterNumber: 212223100036
*/
```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or column.
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.2,random_sta

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
PLACEMENT DATA:

![image](https://github.com/user-attachments/assets/85c724be-fd69-48ea-841e-55b8c0af5fa4)

SALARY DATA:

![image](https://github.com/user-attachments/assets/ae7356cd-6743-4cce-a323-5a8bbdd9978f)

CHECKING THE NULL() FUNCTION:

![image](https://github.com/user-attachments/assets/c6fbf629-76bf-4953-9877-962ac985d251)

DATA DUPLICATE:

![image](https://github.com/user-attachments/assets/abd26f5c-e5c0-4b17-b95b-490b67b39217)

PRINT DATA:

![image](https://github.com/user-attachments/assets/069ba731-e67d-4143-9253-91a0d8181f25)

DATA_STATUS:

![image](https://github.com/user-attachments/assets/619582fc-cdee-4ca2-b33f-9495f2d2aff3)

DATA STATUS:

![image](https://github.com/user-attachments/assets/e5a6f558-01c9-4256-8929-47231caa045b)

Y_PREDICTION ARRAY:

![image](https://github.com/user-attachments/assets/3a25e4e5-d7cc-4cc5-92cc-2aecc107e1c9)

ACCURACY VALUE:

![image](https://github.com/user-attachments/assets/e84eb7e6-9360-4762-9252-cfb3bcd26e0d)

CONFUSION ARRAY:

![image](https://github.com/user-attachments/assets/391fde4e-72e2-446c-a62f-dbd30eea768f)

CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/b9c8d453-b324-4fde-8d0d-b3146b339e68)

PREDICTION OF LR:

![image](https://github.com/user-attachments/assets/add71d88-909d-4512-91a6-dc8d7949ae9e)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
