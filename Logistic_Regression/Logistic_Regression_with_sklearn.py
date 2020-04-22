from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

with open('C:\\Users\\pl\\Downloads\\20news-bydate\\train_processed.txt','r') as f:
	train=f.read().splitlines()
with open('C:\\Users\\pl\\Downloads\\20news-bydate\\test_processed.txt','r') as f:
	test=f.read().splitlines()
X_train,Y_train,X_test,Y_test=[],[],[],[]
for line in train:
	feature=line.split('<fff>')
	X_train.append(feature[-1])
	Y_train.append(feature[0])
for line in test:
	feature=line.split('<fff>')
	X_test.append(feature[-1])
	Y_test.append(feature[0])

tv = TfidfVectorizer(min_df=6)
X_train_Tfidf_vector=tv.fit_transform(X_train)
X_test_Tfidf_vector=tv.transform(X_test)
LR=LogisticRegression(solver='lbfgs',multi_class="multinomial")
LR.fit(X_train_Tfidf_vector,Y_train)
#predicted=LR.predict(X_test_Tfidf_vector)
#lrpp=LR.predict_proba(X_test_Tfidf_vector)
print('Do chinh xac tren tap train :',LR.score(X_train_Tfidf_vector,Y_train))
print('Do chinh xac tren tap test :',LR.score(X_test_Tfidf_vector,Y_test))
