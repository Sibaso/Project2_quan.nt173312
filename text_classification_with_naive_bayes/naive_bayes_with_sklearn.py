
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

tv = TfidfVectorizer(min_df=6)
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
X_train_Tfidf_vector=tv.fit_transform(X_train)
print('kich thuoc thu dien : ',len(tv.get_feature_names()))
mnb=MultinomialNB()
mnb.fit(X_train_Tfidf_vector,Y_train)
X_test_Tfidf_vector=tv.transform(X_test)
predicted=mnb.predict(X_test_Tfidf_vector)
count=0
for i in range(len(Y_test)):
	if predicted[i]==Y_test[i]:
		count+=1
print('so lan du doan dung : '+str(count)+' / '+str(len(Y_test)))
print('precision = ',count/len(Y_test))