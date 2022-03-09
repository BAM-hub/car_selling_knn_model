import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')

x = data[[
    'buying',
    'maint',
    'safety'
]].values
y = data[['class']]
#converting the x
le = LabelEncoder()
for i in range(len(x[0])):
    x[:, i] = le.fit_transform(x[:, i])

#converting y
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)

# create model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn.fit(x_train, y_train)

predictions = knn.predict(x_test)

accuarcy = metrics.accuracy_score(y_test, predictions)

print("predictions", predictions)
print("accuracy", accuarcy)

print('actual value: ', y[20])
print('predicted value', knn.predict(x)[20])