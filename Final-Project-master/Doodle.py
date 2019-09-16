import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#Using google dataset of numpy bitmaps

#testing with single classifier

cat = np.load('full_numpy_bitmap_cat.npy')

X = cat[:15000,:-1].astype('float32')/255 #scale data
y = cat[:15000, -1].astype('float32')

cat = np.c_[cat, np.zeros(len(cat))]

#
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred)
print ('KNN accuracy: ',acc_rf)