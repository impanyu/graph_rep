import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables_by_name
import numpy as np
# this one lets us draw plots in our notebook
#import matplotlib.pyplot as plt
# and we need to do some work with file paths
import os
import tensorflow.contrib.distributions as tfd
import math
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score
import utils
import config


#graph_filename="graphs/ca-netscience.txt"
#graph_filename="graphs/test1.txt"
#embedding_filename="output_ca_netscience.txt"
graph_filename="graphs/web-google.txt"
#graph_filename="graphs/CA-GrQc.txt"
#graph_filename="graphs/test1.txt"
#embedding_filename="output_ca_netscience.txt"
embedding_filename="output.txt"

adj_matrix,vertex_map,edge_count=utils.generate_adj_matrix(graph_filename)
test_edges,test_edges_neg= utils.generate_edges(adj_matrix,len(vertex_map),edge_count)
test_edges.extend(test_edges_neg)

embeddings = utils.read_embeddings(embedding_filename)


score_res = []
for i in range(len(test_edges)):
	score_res.append(np.dot(embeddings[test_edges[i][0]], embeddings[test_edges[i][1]]))
test_label = np.array(score_res)
bar = np.median(test_label) 
ind_pos = test_label >= bar
ind_neg = test_label < bar
test_label[ind_pos] = 1
test_label[ind_neg] = 0
true_label = np.zeros(test_label.shape)
true_label[0:len(true_label) // 2] = 1
		
accuracy = accuracy_score(true_label, test_label)
print(accuracy)









