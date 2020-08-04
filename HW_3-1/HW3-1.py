from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr=LogisticRegression(C=0.01,random_state=0,solver='liblinear',multi_class='auto')
lr_2=LogisticRegression(C=0.1,random_state=0,solver='liblinear',multi_class='auto')
lr_3=LogisticRegression(C=1.0,random_state=0,solver='liblinear',multi_class='auto')
lr_4=LogisticRegression(C=10.0,random_state=0,solver='liblinear',multi_class='auto')
lr_5=LogisticRegression(C=100.0,random_state=0,solver='liblinear',multi_class='auto')
lr_6=LogisticRegression(C=1000.0,random_state=0,solver='liblinear',multi_class='auto')

lr.fit(X_train_std,y_train)
lr_2.fit(X_train_std,y_train)
lr_3.fit(X_train_std,y_train)
lr_4.fit(X_train_std,y_train)
lr_5.fit(X_train_std,y_train)
lr_6.fit(X_train_std,y_train)

y_pred = lr.predict(X_test_std)
y_pred_2 = lr_2.predict(X_test_std)
y_pred_3 = lr_3.predict(X_test_std)
y_pred_4 = lr_4.predict(X_test_std)
y_pred_5 = lr_5.predict(X_test_std)
y_pred_6 = lr_6.predict(X_test_std)




r1=accuracy_score(y_test,y_pred)
r2=accuracy_score(y_test,y_pred_2)
r3=accuracy_score(y_test,y_pred_3)
r4=accuracy_score(y_test,y_pred_4)
r5=accuracy_score(y_test,y_pred_5)
r6=accuracy_score(y_test,y_pred_6)



print("Accurancy:%.2f" % accuracy_score(y_test,y_pred))
print("Accurancy:%.2f" % accuracy_score(y_test,y_pred_2))
print("Accurancy:%.2f" % accuracy_score(y_test,y_pred_3))
print("Accurancy:%.2f" % accuracy_score(y_test,y_pred_4))
print("Accurancy:%.2f" % accuracy_score(y_test,y_pred_5))
print("Accurancy:%.2f" % accuracy_score(y_test,y_pred_6))


