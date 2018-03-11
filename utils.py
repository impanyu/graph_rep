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
import random
import config

def generate_adj_matrix(graph_filename):
	vertex_map={}
	graph_file=open(graph_filename,'r')
	edge_count=0
	i=0
	for line in graph_file:
		eachline=line.split()
		if(int(eachline[0]) not in vertex_map):
			print(eachline[0])
			vertex_map[int(eachline[0])]=i
			i=i+1
		if(int(eachline[1]) not in vertex_map):
			print(eachline[1])
			vertex_map[int(eachline[1])]=i
			i=i+1
		edge_count=edge_count+1
	print(vertex_map)
	graph_file.seek(0)

	V=len(vertex_map)
	adj_matrix=np.zeros(shape=(V,V),dtype='f')
	for line in graph_file:
		eachline=line.split()
		adj_matrix[vertex_map[int(eachline[0])]][vertex_map[int(eachline[1])]]=1

	for i in range(V):
		for j in range(V):
			if(adj_matrix[i][j]==0):
				adj_matrix[i][j]=adj_matrix[j][i]

	return adj_matrix,vertex_map,edge_count


def generate_edges(adj_matrix,V,edge_count):
	#adj_matrix,vertex_map,edge_count=generate_adj_matrix(graph_filename)
	edges=[]
	neg_edges=[]
	#V=len(vertex_map)
	pos=0
	neg=0
	while(True):
	#for x in range(V):
		#for y in range(V):
			x=int(random.random()*V)
			y=int(random.random()*V)
			#print([x,y])
			if(x==y):
				continue
		#print(edge_count//10)
		#print(pos)
		#print(neg)
			if(adj_matrix[x][y]==1 and pos<edge_count//1):
				edges.append([x,y])
				pos=pos+1		
			elif(adj_matrix[x][y]==0 and neg<edge_count//1):
				neg_edges.append([x,y])
				neg=neg+1
			else:
				break
	return edges,neg_edges




def read_embeddings(embeddings_filename):
	embeddings_file=open(embeddings_filename)
	embeddings=[]
	count=0
	for line in embeddings_file:
		eachline=line[1:-2].split()
		if(line[0]!="["):
			for i in range(len(eachline)):
				embeddings[count-1].append(float(eachline[i]))
			continue

		#print(eachline)
		embeddings.append([float(eachline[i]) for i in range(len(eachline))])
		count=count+1
	return embeddings



def read_embeddings_force_direct(embeddings_filename):
	embeddings_file=open(embeddings_filename,'r')
	string=embeddings_file.readline()
	items=string.split()
	vertex_map={}
	node_count=0
	embeddings=[]
	for item in items:
		if(item[0:2]=="cx"):
			current_node=[]
			current_node.append(item[4:-1])
		elif(item[0:2]=="cy"):
			current_node.append(item[4:17])
			vertex_map[current_node[0]+current_node[1]]=node_count
			node_count=node_count+1
			embeddings.append([float(current_node[0]),float(current_node[1])])
	return embeddings,vertex_map,node_count


def read_edges_force_direct(edges_filename,vertex_map,V):
	edges_file=open(edges_filename,'r')
	string=edges_file.readline()
	items=string.split()
	adj_matrix=np.zeros(shape=(V,V),dtype='f')
	edge_count=0
	for item in items:
		if(item[0:2]=="x1"):
			first_node=[]
			first_node.append(item[4:-1])
		elif(item[0:2]=="y1"):
			first_node.append(item[4:17])			
		elif(item[0:2]=="x2"):
			second_node=[]
			second_node.append(item[4:-1])
		elif(item[0:2]=="y2"):
			second_node.append(item[4:17])			
			adj_matrix[vertex_map[first_node[0]+first_node[1]],vertex_map[second_node[0]+second_node[1]]]=1
			edge_count=edge_count+1
	return adj_matrix,edge_count



		
