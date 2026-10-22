# data
X=[[20],[25],[30],[35],[40],[45],[50],[55],[60],[65]] # age(feature)
y=[0,0,0,0,1,1,1,1,1,1] # label(0=young ,1=adult)

# partition data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

# train and apply model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

#evaluation performance
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print("دقت:",acc)