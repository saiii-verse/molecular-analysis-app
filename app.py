import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import Draw

st.set_page_config(
    page_title="Molecular Analysis Lab",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Molecular Analysis Lab")
st.write("Explore molecular structures and chemical properties.")

smiles = st.text_area(
    "Enter molecule SMILES",
    placeholder="Paste a molecule's SMILES here..."
)

if st.button("Analyze Molecule"):
    if not smiles.strip():
        st.warning("Please enter a SMILES string.")
    else:
        mol = Chem.MolFromSmiles(smiles.strip())

        if mol is None:
            st.error("Invalid SMILES. Please check your input.")
        else:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rotatable = Lipinski.NumRotatableBonds(mol)
            atoms = mol.GetNumHeavyAtoms()
            rings = rdMolDescriptors.CalcNumRings(mol)

            st.subheader("🧪 Molecular Structure")
            st.image(Draw.MolToImage(mol, size=(600, 400)))

            st.subheader("📊 Molecular Properties")
            st.write("Molecular Formula:", formula)
            st.write("Molecular Weight:", round(weight, 2), "g/mol")
            st.write("LogP:", round(logp, 2))
            st.write("TPSA:", round(tpsa, 2), "Å²")
            st.write("Hydrogen Bond Donors:", hbd)
            st.write("Hydrogen Bond Acceptors:", hba)
            st.write("Rotatable Bonds:", rotatable)
            st.write("Heavy Atoms:", atoms)
            st.write("Number of Rings:", rings)

            results = pd.DataFrame({
                "Property": [
                    "Formula", "Molecular Weight", "LogP",
                    "TPSA", "H-Bond Donors", "H-Bond Acceptors",
                    "Rotatable Bonds", "Heavy Atoms", "Rings"
                ],
                "Value": [
                    formula, round(weight, 2), round(logp, 2),
                    round(tpsa, 2), hbd, hba, rotatable, atoms, rings
                ]
            })

            st.download_button(
                "Download Results CSV",
                results.to_csv(index=False),
                "molecular_properties.csv",
                "text/csv"
            )
