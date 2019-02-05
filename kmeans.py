
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import statistics

df = pd.read_csv('/home/preetam/iris.csv')
k = ['x','y','z','w']
data = df[k]
print(data['x'].loc[[149]])
'''
def normalize(d):

	for i in ['x','y','z','w']:
		for j in range (0,150):
			d[i].loc[j] = (d[i].loc[j]-d[i].min())/(d[i].max()-d[i].min()) 
	return d
'''
#normalize

data = (data-data.min())/(data.max()-data.min())
print(data)

def euclidean_distance(k,l):
	return ((k[0]-l[0])**2 + (k[1]-l[1])**2 + (k[2]-l[2])**2 + (k[3]-l[3])**2)**(0.5)	




def symmetry_matrix(d):
	data = []
	for i in range(0,150):
		k = []		
		for j in range(0,150):
			k.append(euclidean_distance(d.loc[i],d.loc[j]))	
		data.append(k)
	return data

symm = symmetry_matrix(data)
print(len(symm))

def avg_list(lst):
	return sum(lst)/len(lst)

#print(avg_list(symm[0]))

def form_clusters(matrx):
	clusters = []
	for i in matrx:
		avg = avg_list(i)
		k = []
		for j in range(0,len(i)-1) :
			if i[j] <= avg : 
				k.append(j)
		
		clusters.append(k)
	return clusters

fc = form_clusters(symm)


print(len(fc))



def subset_removal(cluster):
	
	
	for i in range(len(cluster)):
		for j in range(i+1,len(cluster)):
			if j >= len(cluster):
				break
			elif(set(cluster[j]).issubset(cluster[i])):
				del cluster[j]

	print(j," ",len(cluster))
	return cluster
	
	'''
	l2 = cluster[:]

	for m in cluster:
    		for n in cluster:
        		if set(m).issubset(set(n)) and m != n:
            			l2.remove(m)
            			break

			
				
			
	
	return l2
	'''


ll = subset_removal(fc)

#print(len(ll))

def jaccard(a,b):
	x = set(a)
	y = set(b)
	if x!=y:

		k =  len(list(x&y))/len(list(x|y)) 
		return k
	return 0


def symmetry_matrix_clusters(x):
	p = []
	for i in range(0,len(x)):
		k = []
		for j in range(0,len(x)):
			
			k.append(jaccard(x[i],x[j]))
		p.append(k)
	return(p)









def max_value(x):
	
	k  = []
	for i in range(0,len(x)):
		
		k.append([i,x[i].index(max(x[i])),max(x[i])])
	p = []		
	for i in range(0,len(k)):
		p.append(k[i][2])
	t = p.index(max(p))
	s = [k[t][0],k[t][1]]
	return s
	






while True:
	
	jcc = symmetry_matrix_clusters(ll)
	mv = max_value(jcc)
	print(mv)
	un =list(set(ll[mv[0]])|set(ll[mv[1]]))
		
	del ll[mv[0]]
	del ll[mv[1]-1]
	ll.append(un)
	if len(ll)==3:
		break
print(len(ll[0]))
print(len(ll[1]))
print(len(ll[2]))

p = ll[0]
q = ll[1]
r = ll[2]




new_d = [['element',"cluster1","cluster2","cluster3"]]


def distance_avg(medianX,medianY,medianZ,value,no):
	
	dist = []
	l = []
	l.append(euclidean_distance(data.loc[value],data.loc[medianX]))
	l.append(euclidean_distance(data.loc[value],data.loc[medianY]))		
	l.append(euclidean_distance(data.loc[value],data.loc[medianZ]))
	l_sum = sum(l)	
	if no == "all":
		dist.append(value)
		dist.append(euclidean_distance(data.loc[value],data.loc[medianX])/l_sum)
		dist.append(euclidean_distance(data.loc[value],data.loc[medianY])/l_sum)		
		dist.append(euclidean_distance(data.loc[value],data.loc[medianZ])/l_sum)
		new_d.append(dist)		
	elif no == "x&y":
		dist.append(value)
		dist.append(euclidean_distance(data.loc[value],data.loc[medianX])/l_sum)
		dist.append(euclidean_distance(data.loc[value],data.loc[medianY])/l_sum)
		dist.append(1)
		new_d.append(dist)		
	elif no == "y&z":
		dist.append(value)
		dist.append(1)
		dist.append(euclidean_distance(data.loc[value],data.loc[medianY])/l_sum)		
		dist.append(euclidean_distance(data.loc[value],data.loc[medianZ])/l_sum)
		new_d.append(dist)
	elif no=='x&z': 
		dist.append(value)		
		dist.append(euclidean_distance(data.loc[value],data.loc[medianX])/l_sum)		
		dist.append(1)		
		dist.append(euclidean_distance(data.loc[value],data.loc[medianZ])/l_sum)
		new_d.append(dist)
	
	elif no == 'x':
		dist.append(value)
		dist.append(euclidean_distance(data.loc[value],data.loc[medianX])/l_sum)
		dist.append(1)		
		dist.append(1)
		new_d.append(dist)
	elif no == 'y':
		dist.append(value)
		dist.append(1)		
		dist.append(euclidean_distance(data.loc[value],data.loc[medianY])/l_sum)		
		dist.append(1)
		new_d.append(dist)

	elif no == 'z':
		dist.append(value)
		dist.append(1)				
		dist.append(1)
		dist.append(euclidean_distance(data.loc[value],data.loc[medianZ])/l_sum)
		new_d.append(dist)



def plot_elements(a,b,c):
	
	x = set(a)			
	y = set(b)
	z = set(c)

	int_xy = x&y
	int_yz = y&z
	int_xz = x&z
	
	int_xyz = x&y&z	

	only_x = x-int_xy-int_xz
	only_y = y-int_xy-int_yz
	only_z = z-int_xz-int_yz

	only_xy = int_xy-int_xyz
	only_yz = int_yz-int_xyz
	only_xz = int_xz-int_xyz

	only_xyz = int_xyz

	mx = int(statistics.median(x))
	my = int(statistics.median(y))
	mz = int(statistics.median(z))
	
	for i in only_xyz:
		distance_avg(mx,my,mz,i,"all")
	
	for i in only_xy:
		distance_avg(mx,my,mz,i,"x&y")
	
	for i in only_yz:
		distance_avg(mx,my,mz,i,"y&z")
	
	for i in only_xz:
		distance_avg(mx,my,mz,i,"x&z")

	for i in only_x:
		distance_avg(mx,my,mz,i,"x")

	for i in only_y:
		distance_avg(mx,my,mz,i,"y")
	

	for i in only_z:
		distance_avg(mx,my,mz,i,"z")



	
	
	
	color_map = {1:"r",2:"g",3:"b",4:"y",5:"c",6:"k",7:"m"}
	for i in only_y:
		
		plt.scatter(data["x"].loc[i],data["y"].loc[i],color = color_map[1])
	for i in only_z:
		
		plt.scatter(data["x"].loc[i],data["y"].loc[i],color = color_map[2])
	for i in only_xy:
		
		plt.scatter(data["x"].loc[i],data["y"].loc[i],color = color_map[3])
	for i in only_yz:
		
		plt.scatter(data["x"].loc[i],data["y"].loc[i],color = color_map[4])
	for i in only_xz:
		
		plt.scatter(data["x"].loc[i],data["y"].loc[i],color = color_map[5])
	for i in only_xyz:
		
		plt.scatter(data["x"].loc[i],data["y"].loc[i],color = color_map[6])
	
	print ("hey")
	plt.show()
	
	print(mx)
	print(my)
	print(mz)
	




plot_elements(p,q,r)
print(len(new_d))

alpha = []
beta = []
gamma = []

def separate(new_d):
	for i in new_d[1:]:
		minimum_ind = (i.index(min(i)))
		element = i[0]
		if minimum_ind==1:
			alpha.append(element)
		elif minimum_ind==2:
			beta.append(element)
		elif minimum_ind==3:
			gamma.append(element)



separate(new_d)

print(alpha)
print(beta)
print(gamma)


def plotting_function(a,b,c):
	color_map = {1:"r",2:"g",3:"b",4:"y",5:"c",6:"k",7:"m"}	

	for i in a:
		plt.scatter(data["x"].loc[i],data["y"].loc[i],color = color_map[1])				
	for i in b:
		plt.scatter(data["x"].loc[i],data["y"].loc[i],color = color_map[2])				
	for i in c:
		plt.scatter(data["x"].loc[i],data["y"].loc[i],color = color_map[3])				



	plt.show()



plotting_function(alpha,beta,gamma)
























