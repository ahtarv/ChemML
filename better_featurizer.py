import numpy as np  
from rdkit import Chem 

class Featurizer:
    def __init__(self):
        self.known_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I']
        self.known_degrees = [0, 1, 2, 3, 4]  # Fixed: was 'knownn_degrees' (double n)

    def one_hot(self, value, choices):
        """Turns a value into a list of 1s and 0s"""
        encoding = [0] * len(choices)  # Fixed: was 'encoding - [0]' (- instead of =)
        if value in choices:
            encoding[choices.index(value)] = 1 
        return encoding

    def get_matrices(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("invalid SMiles string")

        # Build atom feature matrix
        X = []
        for atom in mol.GetAtoms():
            features = []
            features += self.one_hot(atom.GetSymbol(), self.known_atoms)
            features += self.one_hot(atom.GetTotalDegree(), self.known_degrees)
            features.append(atom.GetFormalCharge())
            X.append(features)

        n_atoms = mol.GetNumAtoms()  # Fixed: was 'GetNmAtoms()' (typo)
        A = np.zeros((n_atoms, n_atoms))  # create an empty nxn grid

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # mark the connection with a 1 (undirected graph means we mark both ways)
            A[i, j] = 1
            A[j, i] = 1

        return np.array(X), A

if __name__ == "__main__":
    f = Featurizer()
    X, A = f.get_matrices("CCO")

    print("Atom Features Matrix (X)")
    print(f"Shape: {X.shape} (3 atoms, {X.shape[1]} features)")

    print("\nAdjacency Matrix")
    print(f"Shape: {A.shape} (3 atoms x 3 atoms)")
    print(A)

    #Adjaceny matrix is basically a mathematical map of the bonds.
    #since neural networks cant see molecules or chemicals they view them through adjaceny matrix
