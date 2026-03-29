import torch 
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd 
import numpy as np 
from rdkit import Chem 

class Featurizer:
    def __init__(self):
        self.known_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I']
        self.known_degrees = [0,1,2,3,4,5]

        self.known_hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ]

    def one_hot(self, value, choices):
        encoding=[0.0] * len(choices)
        if value in choices: encoding[choices.index(value)] = 1.0 
        return encoding 

    def get_matrices(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None 
        n_atoms = mol.GetNumAtoms()
        X=[]
        for atom in mol.GetAtoms():
            feat = self.one_hot(atom.GetSymbol(), self.known_atoms) + \
                    self.one_hot(atom.GetTotalDegree(), self.known_degrees) + \
                    [float(atom.GetFormalCharge())] + \
                    self.one_hot(atom.GetHybridization(), self.known_hybridizations) + \
                    [float(atom.GetIsAromatic())]
            X.append(feat)
        A = np.zeros((n_atoms,n_atoms))
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            A[i,j] = A[j,i] = 1.0 
        np.fill_diagonal(A, 1.0) 
        return torch.tensor(X, dtype=torch.float), torch.tensor(A, dtype=torch.float)

class MolecularGNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gnn1 = nn.Linear(input_dim, 64)
        self.gnn2 = nn.Linear(64,64)
        self.gnn3 = nn.Linear(64,32)
        self.predict_layer = nn.Linear(32,1)

    def forward(self, x, adj):
        h1 = torch.relu(self.gnn1(torch.matmul(adj, x)))
        h2 = torch.relu(self.gnn2(torch.matmul(adj, h1)))
        h3 = torch.relu(self.gnn3(torch.matmul(adj, h2)))
        
        #SUM POOLING: Add all atoms together instead of averaging
        mol_vec = torch.sum(h3, dim=0, keepdim=True)
        
        #Final Prediction
        return self.predict_layer(mol_vec)
df = pd.read_csv('delaney.csv')
f = Featurizer()
dataset = []

print("preparing dataset....")
for _, row in df.iterrows():
    mats = f.get_matrices(row['SMILES'])
    if mats: 
        dataset.append((mats[0], mats[1], torch.tensor([row['measured log(solubility:mol/L)']], dtype=torch.float)))

input_dim = len(f.known_atoms) + len(f.known_degrees) + 1 + len(f.known_hybridizations) + 1
model = MolecularGNN(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nStarting Training....")
for epoch in range(1,51):
    epoch_loss = 0 
    for x, adj, target in dataset: 
        output = model(x, adj)
        optimizer.zero_grad()
        loss=criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch}/50 | Loss: {epoch_loss/len(dataset):.4f}")

test_smiles = "CCO"
x,adj = f.get_matrices(test_smiles)
pred = model(x, adj).item()
print(f"\nPredicted Solubility for Ethanol ({test_smiles}): {pred:.4f}")