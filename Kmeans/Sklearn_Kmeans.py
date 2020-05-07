from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import numpy as np

def load_data(data_path):
	with open(data_path,'r') as f:
		lines=f.read().splitlines();
	labels,tfidf_values,row,col=[],[],[],[]
	for line_id,line in enumerate(lines):
		feature=line.split('<fff>')
		labels.append(int(feature[0]))
		for index_tfidf in feature[2].split():
			tfidf_values.append(float(index_tfidf.split(':')[1]))
			col.append(int(index_tfidf.split(':')[0]))
			row.append(line_id)
	return (tfidf_values,(row,col)),labels

def compute_accuracy(predicted,expected):
	matches=np.equal(predicted,expected)
	accuracy=np.sum(matches)/len(expected)
	return accuracy

def compute_purity(predicted,expected):
  majority_sum=0
  for cluster_index in range(20):
    member_indexs=np.where(predicted==cluster_index)[0]
    expected_labels=[expected[index]for index in member_indexs]
    max_count=max(expected_labels.count(la)for la in range(20))
    majority_sum+=max_count
  print(majority_sum)
  return majority_sum/len(expected)

def clustering_with_Kmeans():
	train_data,train_labels=load_data(
		"C:\\Users\\pl\\Documents\\Python_Project\\ML_DS_2020\\session_1\\TF_IDF\\test_tf_idf_vector.txt")
	test_data,test_labels=load_data(
		"C:\\Users\\pl\\Documents\\Python_Project\\ML_DS_2020\\session_1\\TF_IDF\\test_tf_idf_vector.txt")
	X_train=csr_matrix(train_data)
	X_test=csr_matrix(test_data)
	kmeans=KMeans(n_clusters=20,init='random',n_init=5,tol=1e-3,random_state=2020)
	kmeans.fit(X_train)
	test_predicted=kmeans.predict(X_test)
	priprint('train purity:',compute_purity(kmeans.labels_,train_labels))
	print('test purity:',compute_purity(test_predicted,test_labels))

clustering_with_Kmeans()