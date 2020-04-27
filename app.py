import pandas as pd
from sklearn.tree import DecisionTreeClassifier as dc
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as ac
from sklearn.externals import joblib

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2)

model = dc()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
score = ac(y_test, predictions)
print(score)
joblib.dump(model, 'music-recommender.joblib')
model = joblib.load('music-recommender.joblib')
predictions = model.predict([[32, 1]])
predictions

