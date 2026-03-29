import numpy as np  
from rdkit import Chem  

class Featurizer:
    def __init__(self):
        self.known_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I']
        self.known_degrees = [0,1,2,3,4]
        self.known_bond_types = [Chem.rdchem.BondType.SINGLE,
                                Chem.rdchem.BondType.DOUBLE,
                                Chem.rdchem.BondType.TRIPLE,
                                Chem.rdchem.BondType.AROMATIC]

    def one_hot(self, value, choices):
        """Turns a value into a list of 1s and 0s."""
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        return encoding 
    
    def get_matrices(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Invalid SMILES string")
        
        n_atoms = mol.GetNumAtoms()

        X=[]
        for atom in mol.GetAtoms():
            features = []
            features+= self.one_hot(atom.GetSymbol(), self.known_atoms)
            features+= self.one_hot(atom.GetTotalDegree(), self.known_degrees)
            features.append(atom.GetFormalCharge())
            X.append(features)
        X = np.array(X)

        A = np.zeros((n_atoms, n_atoms))

        n_bond_features = len(self.known_bond_types) + 1
        E = np.zeros((n_atoms, n_atoms, n_bond_features))

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            A[i,j] = A[j,i] = 1

            bond_features = []
            bond_features += self.one_hot(bond.GetBondType(), self.known_bond_types)
            bond_features.append(int(bond.GetIsConjugated()))

            E[i,j] = bond_features
            E[j,i] = bond_features

        return X, A, E

if __name__ == "__main__":
    f = Featurizer()
    X,A,E = f.get_matrices("C=C")

    print(f"Atom Matrix(X) Shape:{X.shape}")
    print(f"Adjaceny Matrix(A) shape:{A.shape}" )
    print(f"Edge Matrix(E) shape:{E.shape}(AatomsxAtomsx Features)\n")

    print("Edge Features between Atom 0 and Atom 1")
    print(E[0,1])
    print("(Format: [Single, Double, Triple, Aromatic, Is_Conjugated])")