import torch
import torch.nn as nn
import numpy as np
from Beter_better_featurizer import Featurizer

f = Featurizer()
X, A, E = f.get_matrices("C=C")

#Assuming X and A come from beter_better_featurizer.py 
x_tensor = torch.tensor(X, dtype=torch.float)
#GNNs usualy prefer a list of edges [2, nums_edges] instead of a full matrix A
edge_index = torch.tensor(np.array(np.where(A==1)), dtype=torch.long)

#a simple gnn layer
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        neighbour_sum = torch.matmul(adj, x) #multiply adjacency matrix by feature matrix
        #this sums up neighbor features for every atom
        #pass the summed features through a standard neural network layer
        return torch.relu(self.linear(neighbour_sum))
    
#run a test pass    
input_size = x_tensor.shape[1] #14 features 
model = SimpleGNN(input_size,32) # output a 32 dimensional embedding
adj_tensor = torch.tensor(A, dtype=torch.float)

output = model(x_tensor, adj_tensor)
print("New Hidden State Shape:", output.shape)
#Result: (2 atoms, 32 learned features )