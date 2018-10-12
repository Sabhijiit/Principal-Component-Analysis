import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt


# reading the dataset into a pandas dataframe
df = pd.read_csv(r"gene_data.csv", header = None)
df = df.dropna(axis=0, how='any')   # removing any na valued rows

# converting type to numpy array and taking transpose
x = df.values 
x = x.T
    
# splitting the dataframe into data X and labels y
X = x[1:,1:]
y = x[1:,0]

# converting entire dataframe to the same numeric data type
X = X.astype('float')

# standardizing
X_std = StandardScaler().fit_transform(X)

# Singular Vector Decomposition
eig_vec, sing_val, nr1 = np.linalg.svd(X_std.T)
eig_val = (sing_val**2)/(X_std.shape[0]-1)

# checking if the norm of eig_vec is 1 which implies it represents axes
for ev in eig_vec:
   np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

# generating (eig_val,eig_vec) pairs
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for 
i in range(len(eig_val))]

# sorting eig_pairs
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# using explained variance to decide what principal components to use
tot = sum(eig_val)
var_exp = [(i/tot)*100 for i in sorted(eig_val, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# combine PCA eig_vec to give eig_vec matrix W
matrix_w = np.hstack((eig_pairs[0][1].reshape(X.shape[1],1), eig_pairs[1][1].reshape(X.shape[1],1)))

# find the projection matrix
Y = X_std.dot(matrix_w)

meta = pd.read_csv(r"Meta data sheet.csv")
meta = meta.ix[:,0:2].values
meta = meta.tolist()

colors = ['k', 'g', [0,1,1], '#9400d3', [1,0,1], '#8b8989', 
'#adff2f', '#ffa500','#ff0000','#32cd32']

count = 0
i = 0

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(20, 20))
    for lab in meta:
        count += 1
        plt.scatter(Y[y==lab[0], 0], Y[y==lab[0], 1], label=lab[0],
                    c=colors[i])
        if count%3 == 0:
            i += 1
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.show()
