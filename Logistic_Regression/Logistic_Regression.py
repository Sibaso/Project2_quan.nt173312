import numpy as np
import json

def load_data_to_sparse(data_path,vocab_size,fname):
	with open(data_path,'r') as f:
		lines=f.read().splitlines()
	X,Y=[],[]
	for line in lines:
		lp=line.split('<fff>')
		label,vector=int(lp[0]),lp[-1]
		dense_vector=[(int(i.split(':')[0]),float(i.split(':')[1])) for i in vector.split()]
		sparse_vecter=[0 for i in range(vocab_size)]
		sparse_vecter.append(1)
		for key,value in dense_vector:
			sparse_vecter[key]=value
		X.append(sparse_vecter)
		Y.append(label)
	with open('C:\\Users\\pl\\Downloads\\20news-bydate\\X_'+fname+'.txt','w') as f:
		json.dump(X,f)
	with open('C:\\Users\\pl\\Downloads\\20news-bydate\\Y_'+fname+'.txt','w') as f:
		json.dump(Y,f)
	return np.array(X),np.array(Y)
		
def sigmoid(x):
	return 1/(1+np.exp(-x))

class Logistic_Regresstion:
	def __init__(self):
		return

	def dot(self,x,w):
		result=w[-1]
		for key,value in x.items():
			result+=w[int(key)]*value
		return result

	def fit(self,X_train,Y_train,learning_rate=2,batch_size=10,max_epoch=50):
		self.labels=set(Y_train)
		self.W=np.array([np.random.randn(X_train.shape[1]) for label in self.labels])
		last_lose=1e9
		for ep in range(max_epoch):
			new_lose=0
			arr=np.array(range(X_train.shape[0]))
			np.random.shuffle(arr)
			X_train=X_train[arr]
			Y_train=Y_train[arr]
			total_batch=int(np.ceil(X_train.shape[0]/batch_size))
			for batch in range(total_batch):
				delta=[[0 for a in range(X_train.shape[1])] for label in self.labels]
				index=batch*batch_size
				X_sub=X_train[index:index+batch_size]
				Y_sub=Y_train[index:index+batch_size]
				dW=[]
				for label in self.labels:
					actual=[]
					for i in Y_sub:
						if label!=i:
							actual.append(0)
						else:
							actual.append(1)
					delta=sigmoid(X_sub.dot(self.W[label]))-actual
					new_lose+=delta.dot(delta)
					dW.append(learning_rate*delta.dot(X_sub))
				dW=np.array(dW)
				self.W=self.W-dW
			new_lose=new_lose/(X_train.shape[0]*len(self.labels))
			print('lose',new_lose)
			if np.abs(last_lose-new_lose)<=1e-4:
			 	print('stop at',ep)
			 	break
			last_lose=new_lose
		return self.W

	def predict(self,x):
		Max,argMax=0,0
		for label in self.labels:
			out=sigmoid(x.dot(self.W[label]))
			if Max<out:
				Max=out
				argMax=label
		return argMax

	def score(self,X_test,Y_test):
		count=0
		for i in range(len(Y_test)):
			vector,label=X_test[i],Y_test[i]
			predicted=self.predict(vector)
			if predicted==label :
				count+=1
		print('so luong du doan dung :',count)
		return count/len(Y_test)
			

# with open('C:\\Users\\pl\\Downloads\\20news-bydate\\words_idf.txt','r') as f:
#   	vocab_size=len(f.read().splitlines())
# X_train,Y_train=load_data_to_sparse('C:\\Users\\pl\\Downloads\\20news-bydate\\train_tf_idf_vector.txt',vocab_size,'train')
# X_test,Y_test=load_data_to_sparse('C:\\Users\\pl\\Downloads\\20news-bydate\\test_tf_idf_vector.txt',vocab_size,'test')
with open('C:\\Users\\pl\\Downloads\\20news-bydate\\X_train.txt','r') as f:
  	X_train=np.array(json.load(f))
with open('C:\\Users\\pl\\Downloads\\20news-bydate\\Y_train.txt','r') as f:
  	Y_train=np.array(json.load(f))
with open('C:\\Users\\pl\\Downloads\\20news-bydate\\X_test.txt','r') as f:
  	X_test=np.array(json.load(f))
with open('C:\\Users\\pl\\Downloads\\20news-bydate\\Y_test.txt','r') as f:
  	Y_test=np.array(json.load(f))
LR=Logistic_Regresstion()
LR.fit(X_train,Y_train,learning_rate=2,batch_size=10,max_epoch=50)
print('Do chinh xac tren tap train :',LR.score(X_train,Y_train))
print('Do chinh xac tren tap test :',LR.score(X_test,Y_test))
