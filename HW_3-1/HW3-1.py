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
test=[0.01,0.1,1,10,100,1000]
acc=[]
for i in test:
    lr=LogisticRegression(C=i,random_state=0,solver="liblinear",multi_class="auto")
    lr.fit(X_train_std,y_train)
    y_pred=lr.predict(X_test_std)
    acc.append(accuracy_score(y_test,y_pred))
acc=np.round(acc,2)

plt.plot(range(-2,4),acc)
plt.title("Logistic Regression")
plt.xlabel("c:10^")
plt.ylabel("Accuracy")
plt.grid("on")
plt.show()

