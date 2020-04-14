import numpy as np
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer=PorterStemmer()
stop_words=set(stopwords.words('english'))


def gather_data(data_path):
	with open(data_path,'r') as f:
		lines=f.read().splitlines()
	X,Y=[],[]
	for line in lines:
		X.append(line.split('<fff>')[-1])
		Y.append(line.split('<fff>')[0])
	return X,Y


class naive_bayes:

	bag_of_words=set()
	log10_word_in_label=dict()
	log10_label=dict()

	def __init__(self):
		with open('C:\\Users\\pl\\Downloads\\20news-bydate\\words_idf.txt','r')as f:
			self.bag_of_words=set([line.split('<fff>')[0] for line in f.read().splitlines()])
		with open('C:\\Users\\pl\\Downloads\\20news-bydate\\log10_word_in_label.txt','r') as f:
			self.log10_word_in_label=dict([((line.split('<fff>')[0],line.split('<fff>')[1]),float(line.split('<fff>')[2])) for line in f.read().splitlines()])
		with open('C:\\Users\\pl\\Downloads\\20news-bydate\\log10_label.txt','r') as f:
			self.log10_label=dict([(line.split('<fff>')[0],float(line.split('<fff>')[1])) for line in f.read().splitlines()])
		return

	def fit(self,X_train,Y_train):
		count_word_in_label=defaultdict(int)
		total_word_in_label=defaultdict(int)
		count_doc_in_label=defaultdict(int)
		labels=set()
		for i in range(len(Y_train)):
			label,doc=Y_train[i],X_train[i]
			labels.add(label)
			count_doc_in_label[label]+=1
			for word in doc.split():
				if word in self.bag_of_words:
					count_word_in_label[(label,word)]+=1
					total_word_in_label[label]+=1
		for label in labels:
			self.log10_label[label]=np.log10(count_doc_in_label[label]/len(Y_train))
			for word in self.bag_of_words:
				self.log10_word_in_label[(label,word)]=np.log10((count_word_in_label[(label,word)]+1)/(total_word_in_label[label]+len(self.bag_of_words)))
		with open('C:\\Users\\pl\\Downloads\\20news-bydate\\log10_word_in_label.txt','w') as f:
			f.write('\n'.join([label+'<fff>'+word+'<fff>'+str(value) for (label,word),value in self.log10_word_in_label.items()]))
		with open('C:\\Users\\pl\\Downloads\\20news-bydate\\log10_label.txt','w') as f:
			f.write('\n'.join([label+'<fff>'+str(value) for label,value in self.log10_label.items()]))

	def transform(self,X):
		return [list(set([word for word in doc.split() if word in self.bag_of_words])) for doc in X]


	def predict(self,X_test):
		predicted=[]
		for vector in X_test:	
			Max,arg_max=-1e9,0
			for label in self.log10_label.keys():
				t=self.log10_label[label]
				for word in vector:
					t+=self.log10_word_in_label[(label,word)]
				if Max<t:
					Max=t
					arg_max=label
			predicted.append(arg_max)
		return predicted


X_train,Y_train=gather_data('C:\\Users\\pl\\Downloads\\20news-bydate\\train_processed.txt')
X_test,Y_test=gather_data('C:\\Users\\pl\\Downloads\\20news-bydate\\test_processed.txt')
nabe=naive_bayes()
#nabe.fit(X_train,Y_train)
X_test_vector=nabe.transform(X_test)
predicted=nabe.predict(X_test_vector)
count=0
for i in range(len(Y_test)):
	if predicted[i]==Y_test[i]:
		count+=1
print('so lan du doan dung : '+str(count)+' / '+str(len(Y_test)))
print('precision = ',count/len(Y_test))