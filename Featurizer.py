from rdkit import Chem
import numpy as numpy
class Featurizer:
    def __init__(self):
        pass
    def get_features(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return "Invalid SMILES"
        
        atom_features = []
        for atom in mol.GetAtoms():
            features = {
                "symbol": atom.GetSymbol(),
                "atomic_num": atom.GetAtomicNum(),
                "chirality": str(atom.GetChiralTag()),
                "degree": atom.GetTotalDegree(),
                "formal_charge": atom.GetFormalCharge(),
            }
            atom_features.append(features)

        bond_features = []
        for bond in mol.GetBonds():
            features = {
                "indices": [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                "type": str(bond.GetBondType()),
                "is_conjugated": bond.GetIsConjugated(),
            }
            bond_features.append(features)

        return{"atoms": atom_features, "bonds": bond_features}

f = Featurizer()
molecule_data = f.get_features("CCO")

print("Atom Features")
for i, atom in enumerate(molecule_data['atoms']):
    print(f"Atom: {i}: {atom}")

print("\nBond Features")
for i, bond in enumerate(molecule_data['bonds']):
    print(f"Bond: {i}: {bond}")