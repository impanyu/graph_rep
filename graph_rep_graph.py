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
import config
from utils import *

#a=np.zeros([3])
#print(a)
#np.expand_dims(a,axis=1)
#a[np.newaxis,:]

#outputfile="inputs/splitgraph"+sys.argv[1]+".txt"
#inputfile="graphs/test1.txt"
inputfile="graphs/karate.txt"

#inputfile="graphs/ca-netscience.txt"
#inputfile="eco-florida.txt"
#inputfile="graphs/web-google.txt"
#inputfile="graphs/frb30-15-1.txt"
#inputfile="graphs/email-Eu-core.txt"
#reversemap="inputs/reversemap"+sys.argv[1]+".txt"
#n=sys.argv[3]
#inputfile="graphs/CA-GrQc.txt"


infile =open(inputfile,'r')
outfile=open("output.txt",'w')
outfile2=open("output2.txt",'w')
#reversemapfile=open(reversemap,'w')
#E=8
#V=16
V=34
#V=379
#V=450
#V=1005
#V=1299
#V=5242
number_of_component=2
#E=25571*2
#E=2773*2
#E=14496

#V=5242
#E=2010
E=156
iter_n=6500
#E=30
#E=914*2
#chosen_lines=np.sort((lines_number*np.random.rand(lines_number/int(n))).astype(int))
#print chosen_lines
previous=-1
i=0
j=1
pi=0
pj=0
p=0
vertex_map={}
degree=[]
max_degree=0

init_span=1.0



adj_list,vertex_map,edge_count=generate_adj_list(inputfile)
print(adj_list)
ds=[]
for i in range(V):
	ds.append(len(adj_list[i]))

dimensions=config.dimensions

embeddings=tf.get_variable("embeddings",[V,dimensions],initializer=tf.random_normal_initializer(.1))
v=tf.get_variable("sigma",[],initializer=tf.random_normal_initializer(.5))
line_count=0

all_ones=tf.ones([V])


np.random.seed(11)



new_embeddings=[]
last_embeddings=[]
distances=[]
layer_n=4
randoms={}

for i in range(V):
	randoms[i]={}
	for j in range(ds[i]):
		if(i<adj_list[i][j]):
			randoms[i][adj_list[i][j]]=(tf.random_normal([dimensions],0,0.5))
		else:
			randoms[i][adj_list[i][j]]=(-randoms[adj_list[i][j]][i])



#random_v=np.random.normal(0,0.5,size=(V))
#append first layer
#new_embeddings.append(embeddings)

#for l in range(layer_n-1):
#	new_embeddings.append([])
#node_order=np.random.permutation(V)
#print(node_order)
for l in range(layer_n):
	if(l==0):
		new_embeddings.append(embeddings)#+tf.random_normal([V,dimensions],0,0.6))
		
	else:
		new_embeddings.append([])
		for i in range(V):
			new_embeddings[l].append(0)
			for j in range(ds[i]):
				#outfile.write(str(i)+" "+str(adj_list[i][j])+"\n")
				if(l==1):
					new_embeddings[l][i]=new_embeddings[l][i]+new_embeddings[l-1][adj_list[i][j]]+randoms[i][adj_list[i][j]]#+randoms[i][j]#+tf.random_normal([dimensions],0,0.6))#+np.random.normal(0,1/ds[j],size=dimensions)
				else:
					new_embeddings[l][i]=new_embeddings[l][i]+new_embeddings[l-1][adj_list[i][j]]+randoms[i][adj_list[i][j]]#+tf.random_normal([dimensions],0,0.6)#+np.random.normal(0,1/ds[j],size=dimensions)
				#distances.append (tf.reduce_sum(tf.square(embeddings[i]-embeddings[j])))
			
			#new_embeddings[l].append(tf.reduce_sum(tf.gather(new_embeddings[l-1],adj_list[i]),0))
			if(ds[i]!=0):
				new_embeddings[l][i]=new_embeddings[l][i]/ds[i]#+tf.random_normal([dimensions],0,1/ds[i])#   +np.random.normal(0,1/ds[i],size=dimensions)
			#if(l == layer_n-1):
			#	last_embeddings.append(tf.identity(new_embeddings[l][i]))

				#new_embeddings[l][i]=new_embeddings[l][i]+tf.random_normal([dimensions],0,0.6)

#difference=tf.tensordot(tf.reduce_sum((new_embeddings[layer_n-1]-new_embeddings[0])**2,1),tf.cast(ds,tf.float32),[[0],[0]])/E

difference=0
'''
for l in range(layer_n):
	#diff=0
	#for i in range(V):
		#diff=diff+ tf.reduce_sum((new_embeddings[layer_n-1-l][i]-new_embeddings[l][i])**2)
	difference=difference+tf.reduce_mean((embeddings-new_embeddings[l])**2)#+tf.reduce_mean(distances)
	#difference=difference+diff/V
difference=difference/layer_n
'''
mean=tf.reduce_mean(embeddings,0)

var=tf.tensordot(embeddings-mean,embeddings-mean,[[0],[0]])/V
difference=tf.reduce_mean((embeddings-new_embeddings[layer_n-1])**2/[var[0][0],var[1][1]])

#difference=tf.tensordot( tf.reduce_mean((embeddings-new_embeddings[layer_n-1])**2/[var[0][0],var[1][1]],1),tf.cast(ds,tf.float32),[[0],[0]])/tf.reduce_sum(tf.cast(ds,tf.float32))
difference_total=difference#+(tf.abs(var[0][1])+tf.abs(var[1][0]))/2
                    

optimizer = tf.train.AdamOptimizer(learning_rate=.001)
train_op = optimizer.minimize(difference_total)

session=tf.Session()
#session=tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run(tf.global_variables_initializer())

#print(mu_indices.eval(session=session))
i=0
for m in range(iter_n): #train iter_n times
    r=session.run(train_op)  
    i=i+1
    if(i%100==0): 
        print(difference.eval(session=session))
        print(i)

ebs=embeddings.eval(session=session)

print(v.eval(session=session))
for i in range(V):
    #print(ebs[i])
    outfile.write(str(ebs[i])+"\n")
    outfile2.write(str(new_embeddings[layer_n-1][i].eval(session=session))+"\n")
#reversemapfile.close()		
infile.close()
