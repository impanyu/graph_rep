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

#a=np.zeros([3])
#print(a)
#np.expand_dims(a,axis=1)
#a[np.newaxis,:]

#outputfile="inputs/splitgraph"+sys.argv[1]+".txt"
#inputfile="graphs/test1.txt"
#inputfile="graphs/karate.txt"

#inputfile="graphs/ca-netscience.txt"
#inputfile="eco-florida.txt"
inputfile="graphs/web-google.txt"
#inputfile="graphs/frb30-15-1.txt"
#inputfile="graphs/email-Eu-core.txt"
#reversemap="inputs/reversemap"+sys.argv[1]+".txt"
#n=sys.argv[3]
#inputfile="graphs/CA-GrQc.txt"


infile =open(inputfile,'r')
outfile=open("output.txt",'w')
#reversemapfile=open(reversemap,'w')
#E=8
#V=16
#V=34
#V=379
#V=450
#V=1005
V=1299
#V=5242
number_of_component=2
#E=25571*2
E=2773*2
#E=14496

#V=5242
#E=2010
#E=156
iter_n=2000
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
adj_matrix=np.zeros(shape=(V,V),dtype='f')
adj_matrix_cond=-np.ones(shape=(V,V),dtype='f')
full_matrix=np.zeros(shape=(V,V,config.dimensions),dtype='f')

#reverse_degree=[]

#init_span=.58#0.1*tf.sqrt(float(V))
init_span=1.0
'''
for vertex in chosen_lines:
	vertexmap[str(vertex)]=str(i)
        reversemapfile.write(str(i)+" "+str(vertex)+"\n")
        i=i+1
'''
#infile.seek(0)

for line in infile:
	eachline=line.split()
	if(int(eachline[0]) not in vertex_map):
		print(eachline[0])
		vertex_map[int(eachline[0])]=i
		i=i+1
	if(int(eachline[1]) not in vertex_map):
		print(eachline[1])
		vertex_map[int(eachline[1])]=i
		i=i+1
print(vertex_map)

'''
	if (int(eachline[0]) !=previous):
		if(i!=-1):
			degree.append(j)
		i=i+1
		vertex_map[int(eachline[0])]=i
		if(j>max_degree):
			max_degree=j
		j=1
	else:
		j=j+1
	previous=int(eachline[0])
'''
degree.append(j)
if(j>max_degree):
	max_degree=j


infile.seek(0)

#initialize the adjacency matrix
for line in infile:
	eachline=line.split()
	adj_matrix[vertex_map[int(eachline[0])]][vertex_map[int(eachline[1])]]=1 
	adj_matrix_cond[vertex_map[int(eachline[0])]][vertex_map[int(eachline[1])]]=1
	full_matrix[vertex_map[int(eachline[0])]][vertex_map[int(eachline[1])]][0]=1 
	full_matrix[vertex_map[int(eachline[0])]][vertex_map[int(eachline[1])]][1]=1

adj_matrix_copy=np.copy(adj_matrix)
for i in range(V):
	for j in range(V):
		if(adj_matrix[i][j]==0):
			adj_matrix[i][j]=adj_matrix[j][i]
			full_matrix[i][j][0]=adj_matrix[j][i] 
			full_matrix[i][j][1]=adj_matrix[j][i]
		if(i==j):
			adj_matrix_copy[i][j]=1


adj_matrix=np.transpose(adj_matrix)
degrees=tf.reduce_sum(adj_matrix,1)
max_degree=tf.reduce_max(degrees)
mean_degree=tf.reduce_mean(degrees)
degree_ratio=degrees/V
degree_ratio_square=tf.multiply(tf.square(degrees),degree)/V
degree_ratio_sqrt=tf.sqrt(degree_ratio)
reverse_degree=max_degree/(degrees+.1)

reverse_d=tf.tensordot(adj_matrix,reverse_degree,1)

#for i in range(V):
#	reverse_degree.append(max_degree/degree[i] if degree[i]>0 else 0)

infile.seek(0)
#location=tf.Variable(tf.truncated_normal([3]), name='location')
#scale=tf.Variable(tf.truncated_normal([3]), name='scale'
#mvn=[]
#V=5
dimensions=config.dimensions
mus=[]
sigmas=[]


values,mu_indices=tf.nn.top_k(degrees,k=number_of_component)
embeddings=tf.get_variable("embeddings",[V,dimensions],initializer=tf.random_normal_initializer(.2))
embeddings_hub=tf.gather(embeddings,mu_indices)


#for i in range(1):
#    mus.append(tf.get_variable("mus"+str(i),[dimensions],initializer=tf.random_uniform_initializer(-init_span,init_span)))
#    sigmas.append(tf.get_variable("sigmas"+str(i),[],initializer=tf.random_normal_initializer(init_span,0.2)))
    #mus.append(embeddings[mu_indices[i]])

'''
for i in range(number_of_component):
    location.append(tf.Variable(tf.truncated_normal([dimensions])))
    #scale.append(tf.Variable(tf.contrib.distributions.matrix_diag_transform(tf.truncated_normal([3,3]), transform=tf.nn.softplus)))
    scale.append(tf.Variable(tf.truncated_normal([1]))*tf.ones([dimensions]))
    mvn.append(tfd.MultivariateNormalDiag(
          loc=location[i],
          scale_diag=scale[i]))
'''
    
#mix_gauss = tfd.Mixture(
#  cat=tfd.Categorical(probs=tf.nn.softmax(probs)),
#  components=mvn)


#x = tf.placeholder(tf.float32, [V, dimensions], name='data')
#print(location.eval(session=session))
#e=tf.truncated_normal([dimensions])
#embeddings=[]

#for i in range(V):
#    embeddings.append(tf.Variable(e))
cond_sigmas=tf.get_variable("condsigmas",[V],initializer=tf.random_uniform_initializer(0,1))


#ratios=tf.get_variable("ratios",[V],initializer=tf.random_uniform_initializer(0,1))
#ratios=tf.nn.softmax(tf.get_variable("ps",[V],initializer=tf.random_uniform_initializer))



#for i in range(V):
#    conditionalDist.append(tfd.MultivariateNormalDiag(
#                  loc=embeddings[i],:1000:"::200
#                  scale_diag=conditional_scale*mix_gauss.prob(embeddings[i])))
#    if i%10==0:
#o        print(i)
        
#loglikelihood=conditionalDist[0].log_prob(embeddings[0])

#for i in range(V):
#    print(embeddings[i].eval(session=session))
#matrix=[[0,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]]

line_count=0

#loglikelihood=tf.tensordot(tf.reduce_sum(adj_matrix,1),prior_dist,1)

all_ones=tf.ones([V])
cond_dists=[]


'''
for line in infile:
	eachline=line.split()
	i=vertex_map[int(eachline[0])]
	j=vertex_map[int(eachline[1])]
	conditionalDist= tfd.MultivariateNormalDiag(loc=embeddings[i],scale_diag=conditional_scale*mix_gauss.prob(embeddings[i]))
	if(line_count==0):
		loglikelihood= int(max_degree/degree[i])* (locality*conditionalDist.log_ prob(embeddings[j]) + mix_gauss.log_prob(embeddings[j]))

	else:
    
		loglikelihood+= int(max_degree/degree[i])* (locality*conditionalDist.log_prob(embeddings[j]) + mix_gauss.log_prob(embeddings[j]))
	if(line_count%10==0):
		print(line_count)
	line_count=line_count+1

for i in range(V):
    loglikelihood+=mix_gauss.log_prob(embeddings[i])
'''

embeddings_adjusted=tf.transpose(tf.transpose(embeddings)*reverse_d)
#embeddings_mean=tf.reduce_sum(embeddings_adjusted,0)/V/max_degree

embeddings_means=tf.matmul(adj_matrix,embeddings)/V
embeddings_means_max=tf.reduce_max(tf.abs(embeddings_means-embeddings))
embeddings_primes=[]
embeddings_primes.append(adj_matrix*embeddings[:,0]-embeddings_means[:,0])
embeddings_primes.append(adj_matrix*embeddings[:,1]-embeddings_means[:,1])
#embeddings_primes=tf.stack(embeddings_primes,axis=2)-embeddings_means


embeddings_mean=tf.reduce_mean(embeddings,0)
embeddings_mean_max=tf.reduce_mean(tf.abs(embeddings_mean))
embeddings_prime=embeddings-embeddings_mean


#embeddings_prime=tf.transpose(tf.transpose(embeddings-embeddings_mean)*reverse_d)
#embeddings_p=tf.transpose(tf.transpose(embeddings-embeddings_mean))
#embeddings_prime2=tf.transpose(tf.transpose(embeddings_prime)*reverse_d)

np.random.seed(40)
#embeddings_var=[[0,0],[0,0]]
s=np.random.permutation(V)
#print(s[0:10])
#embeddings_var=[]

#probs=tf.nn.softmax(tf.get_variable("probs",[number_of_component],initializer=tf.random_uniform_initializer))
new_embeddings=[]
#embeddingsp=[]

random_signals=np.random.normal(0,1,(V,dimensions))
for i in range(V):
	#embeddingsp.append(embeddings[i])
	new_embeddings.append(embeddings[i])
#print(new_embeddings.get_shape())
#new_embeddings=tf.transpose(tf.transpose(tf.reduce_sum(full_matrix*embeddings,1))/degrees)+random_signals
for i in range(V):
	first=True
	for j in range(V):
		if(adj_matrix[i][j]==1):
			if(first):
				new_embeddings[i]=new_embeddings[j]
				first=False
			else:
				new_embeddings[i]=new_embeddings[i]+new_embeddings[j]
	if(first!=True):
		new_embeddings[i]=(new_embeddings[i]/degrees[i]+np.random.normal(0,.05,(dimensions)))
	

#tf.tensordot(adj_matrix[i],new_embeddings,axes=[[0],[0]])/degrees[i]+np.random.normal(0,.05,(2))


difference=tf.reduce_sum((embeddings-new_embeddings)**2)/V


embeddings_each_vars=[]
#full_matrix=tf.expand_dims(adj_matrix,2)
embeddings_each_vars=tf.transpose(full_matrix*embeddings,perm=[1,0,2])-embeddings
embeddings_each_vars=tf.transpose(embeddings_each_vars,perm=[1,0,2])*full_matrix
#embeddings_each_vars=embeddings_each_vars

embeddings_each_vars=tf.reduce_sum(embeddings_each_vars*embeddings_each_vars,2)
embeddings_each_vars_reverse=(embeddings_each_vars)

embeddings_each_vars_reverse=tf.pow(math.e,-embeddings_each_vars_reverse*12)*adj_matrix
embeddings_each_vars_sum=tf.reduce_sum(embeddings_each_vars_reverse,1)
embeddings_each_vars_max=tf.reduce_max(embeddings_each_vars,1)


embeddings_each_vars_ratio=tf.transpose(tf.transpose(embeddings_each_vars_reverse)/embeddings_each_vars_sum)

degrees_neighbour=tf.tensordot(adj_matrix,degrees,1)

prior_dist=[]
p=tf.get_variable("p",[],initializer=tf.random_normal_initializer(0,.5))
b=tf.get_variable("b",[],initializer=tf.random_normal_initializer(0,.5))

for i in range(V):
	#prior_dist.append(0.1)
	prior_dist.append((degrees[i]+0*degrees_neighbour[i])/(embeddings_each_vars_max[i]*2*math.pi)/V)



#for i in range(V):
#	prior_dist[i]=tf.tensordot(embeddings_each_vars_ratio[i],prior_dist,1) 

#prior_dist=prior_dist/degrees

#embeddings_each_vars=tf.sqrt(embeddings_each_vars)

'''
k=V
embeddings_each_topd,top_indices=tf.nn.top_k(-tf.reduce_sum(embeddings_each_vars,2),k)
#embeddings_each_vars=-embeddings_each_topd
my_range = tf.expand_dims(tf.range(0, V), 1)  # will be [[0], [1]]
my_range_repeated = tf.tile(my_range, [1, k]) #[[0,0],[1,1]]
full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(top_indices, 2)], axis=2) #[V,k,2]



embeddings_each_vars=tf.gather_nd(embeddings_each_vars,full_indices)
embeddings_each_v=tf.reduce_mean(embeddings_each_vars[:,:,0]*embeddings_each_vars[:,:,1],1)
embeddings_each_vars=tf.reduce_mean(embeddings_each_vars,1)
embeddings_each_vars=tf.reduce_min(embeddings_each_vars,1)#-embeddings_each_v
embeddings_p=1/embeddings_each_vars/math.pi/2
embeddings_p=(1/2/tf.sqrt(embeddings_each_vars)-embeddings_p)/embeddings_each_vars*tf.abs(embeddings_each_v)+embeddings_p
'''

'''

embeddings_mean=tf.reduce_mean(embeddings,0)
embeddings_mean_max=tf.reduce_max(tf.abs(embeddings_mean))
embeddings_prime=embeddings-embeddings_mean
embeddings_var=tf.tensordot(embeddings_prime,embeddings_prime,[[0],[0]])/V

#embeddings_vars=[]
#embeddings_vars.append(tf.abs(tf.reduce_sum(embeddings_primes[0]*embeddings_primes[0],1)/degrees))
#embeddings_vars.append(tf.abs(tf.reduce_sum(embeddings_primes[1]*embeddings_primes[1],1)/degrees))
#embeddings_vars.append(tf.abs(tf.reduce_sum(embeddings_primes[0]*embeddings_primes[1],1)/degrees))
#embeddings_vars=tf.reduce_max(embeddings_vars,0)

#embeddings_var=embeddings_var/float(V*8)
#embeddings_hub_mean=tf.reduce_mean(embeddings_hub,0)
#embeddings_hub_prime=embeddings_hub-embeddings_hub_mean
#embeddings_hub_var=tf.tensordot(embeddings_hub_prime,embeddings_hub_prime,[[0],[0]])/number_of_component




#min_var=embeddings_var[0,0]+embeddings[1,1]-tf.abs(embeddings[0,1])
#mean_var=tf.reduce_mean([embeddings_var[0,0],embeddings[1,1]])
dists=tf.contrib.distributions.MultivariateNormalDiag(embeddings_mean,[tf.sqrt(embeddings_var[0,0]),tf.sqrt(embeddings_var[1,1])])#[embeddings_var[0,0],embeddings_var[1,1]])




#prior_dist=embeddings_each_vars
#for i in range(number_of_component):
#prior_dist=dists.prob(embeddings)


#prior_dist=tf.reduce_sum(prior_dist,0)
#print(prior_dist.get_shape())
print("complete building prior distrition")


#conditional_scale=tf.Variable(tf.contrib.distributions.matrix_diag_transform(tf.truncated_normal([3]), transform=tf.nn.softplus))
#stacked_sigmas=[]
#for i in range(dimensions):
#	stacked_sigmas.append(prior_dist)


#p=tf.Variable(1.)

#all_cond_sigmas=tf.transpose(tf.stack(stacked_sigmas))/3+0.1
#all_cond_sigmas=all_cond_sigmas)*100
#locality=tf.Variable(1.)
#conditionalDist=[]
#s=tf.Variable(100.0)


#embeddings_each_vars=(1/3/(embeddings_each_vars+.001)**2-1)*tf.abs(embeddings_each_v)+embeddings_each_vars  #embeddings_each_vars-tf.abs(embeddings_each_v)

s=np.random.permutation(V)
sample_length=int(V*0.2)
sampled_embeddings=tf.gather(embeddings_prime,s[:sample_length])
sampled_histogram=np.zeros(shape=(sample_length),dtype=float)

for i in range(sample_length):
	sampled_histogram[tf.to_int32(sampled_embeddings[i][0]/init_span/3)+tf.to_int32(sampled_embeddings[i][1]/init_span/3)]+=1

sampled_histogram=sampled_histogram/sampled_length

grid_embeddings=[]
for i in range(10):
	for j in range(10):
		grid_embeddings.append([init_span*3/10*i,init_span*3/10*j])


grid_prob=dists[0].prob(grid_embeddings)

similarity=tf.reduce_mean(tf.abs(sampled_histogram-grid_prob))
'''

#for j in range(100):
#      s=np.random.permutation(V)

#embeddings_var.append(tf.tensordot(tf.gather(embeddings_prime,s[:int(V*0.8)]),tf.gather(embeddings_prime,s[:int(V*0.8)]),[[0],[0]])/V/0.8)
#print(tf.__version_)

'''
for i in range(V):
	#cond_dist = tf.contrib.distributions.MultivariateNormalDiag(embeddings[i],[.1,.1])
	cond_dist = tf.contrib.distributions.MultivariateNormalDiag(embeddings[i],[prior_dist[i]+0.05,prior_dist[i]+0.05])#[10/(embeddings_vars[i]+0.1)/math.pi/2,10/(embeddings_vars[i]+.1)/math.pi/2]) #all_cond_sigmas[i])
	#cond_dist = tf.contrib.distributions.MultivariateNormalDiag(embeddings[i],[0.5,0.5])#[10/(embeddings_vars[i]+0.1)/math.pi/2,10/(embeddings_vars[i]+.1)/math.pi/2]) #all_cond_sigmas[i])
	
	#cond_dists.append(tf.reduce_sum(tf.tensordot(adj_matrix[i], cond_dist.log_prob(embeddings),1))k)
	#cond_dists.append(tf.tensordot(tf.multiply(cond_dist.log_prob(embeddings),degree_ratio) ,adj_matrix_cond[i],1))
	cond_dists.append( tf.tensordot( adj_matrix[i],cond_dist.log_prob(embeddings),1 ))#*reverse_degree[i])
	print(i)
'''


#sigma_avg=1/tf.sqrt(number_of_component)
#for i in range(1):
#	if(i==0):
#		sigma_dif=tf.nn.relu(0.1-sigmas[i])
#	else:
#		sigma_dif=sigma_dif+tf.nn.relu(0.1-sigmas[i])

#sigma_dif=tf.nn.relu(sigma_dif)

#max_value=tf.reduce_max(tf.abs(embeddings))

#embedding_max=tf.nn.relu(max_value-init_span*3)

#loglikelihood=tf.reduce_sum(cond_dists)/E#+tf.reduce_sum(prior_dist)/V#+tf.tensordot(reverse_d,prior_dist,1) 
    
#loglikelihood=tf.reduce_mean(cond_dists)
#loglikelihood = tf.reduce_mean(tf.tensordot(cond_dists,prior_dist,1))
#loglikelihood = -loglikelihood/V-tf.abs(sigmas[0]+sigmas[1]-2)*100
#var_dif=tf.nn.relu(tf.reduce_mean ([1.0,1.0]-tf.diag_part(tf.tensordot(embeddings_prime,embeddings_prime,[[0],[0]])/V) ) )
var_dif=[]
#for j in range(100):
#var_dif=tf.reduce_max(tf.abs (tf.diag([1.0 for i in range(dimensions)])-embeddings_var) )

#var_dif_mean=tf.reduce_max(var_dif)
#var_dif2=tf.reduce_mean(tf.abs ([1.0 for i in range(dimensions)]-tf.diag_part(embeddings_var)) )
#var_hub=tf.reduce_mean(tf.abs (tf.diag([1.0 for i in range(dimensions)])-embeddings_hub_var) )


#loglikelihood = -loglikelihood#/E#/max_degree/V
#regularizer=var_dif+embeddings_means_max

#p=tf.get_variable("coefficient",[],initializer=tf.random_normal_initializer(0,1))
#regularizer=(tf.sqrt(tf.abs(embeddings_var[0,1])))#+tf.abs(tf.sqrt(embeddings_var[0,0])-tf.sqrt(embeddings_var[1,1]))) #5*(var_dif+embeddings_mean_max)#(0*embedding_max+0*embeddings_vars+0*embeddings_means_max+var_dif)*5#++embeddings_mean_max) #+300*var_hub
#regularizer=similarity*10



#loglikelihood=loglikelihood+regularizer*5
#+100*tf.abs(tf.reduce_mean(tf.tensordot(embeddings_prime,embeddings_prime,[[0],[0]])/V-tf.diag([1.,1.])))+sigma_dif*100 #+0.5*tf.reduce_mean(embeddings_mean)

optimizer = tf.train.AdamOptimizer(learning_rate=.01)
train_op = optimizer.minimize(difference)

session=tf.Session()
#session=tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run(tf.global_variables_initializer())

print(mu_indices.eval(session=session))
i=0
for m in range(iter_n): #train iter_n times
    r=session.run(train_op)  
    #print(degree_ratio[0].eval(session=session))
    #print(embeddings[0].eval(session=session))
    i=i+1
    if(i%100==0): 
        print(difference.eval(session=session))
        #print(regularizer.eval(session=session))
        print(i)
    #print(loglikelihood.eval(session=session))
    #print(all_cond_sigmas[0].eval(session=session)) 
    #print(prior_dist[0].eval(session=session))
    #print(prior_dist[1].eval(session=session))

#session.run([train_op,embeddings])

#print(locality.eval(session=session))
#for i in range(1):
#	print(sigmas[i].eval(session=session))
#print(sigmas[1].eval(session=session))
#print(var_dif.eval(session=session))
#print(prior_dist.eval(session=session))
#print(p.eval(session=session))
ebs=embeddings.eval(session=session)
for i in range(V):
    #print(ebs[i])
    outfile.write(str(ebs[i])+"\n")
#reversemapfile.close()		
infile.close()
