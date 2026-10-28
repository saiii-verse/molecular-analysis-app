
 import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import Draw

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MolecuLab Pro",
    page_icon="🧬",
    layout="wide"
)

# ---------------- MOLECULE LIBRARY ----------------

MOLECULES = {
    "Aspirin": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "category": "Drug"
    },
    "Paracetamol": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "category": "Drug"
    },
    "Ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "category": "Drug"
    },
    "Caffeine": {
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "category": "Drug"
    },
    "Amoxicillin": {
        "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
        "category": "Drug"
    },
    "Metformin": {
        "smiles": "CN(C)C(=N)NC(=N)N",
        "category": "Drug"
    },
    "Omeprazole": {
        "smiles": "COc1ccc2nc(sc2c1)S(=O)Cc1ncc(C)c(OC)c1C",
        "category": "Drug"
    },
    "Atorvastatin": {
        "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)nc2cc(-c3ccccc3)nn12",
        "category": "Drug"
    },
    "Ciprofloxacin": {
        "smiles": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "category": "Drug"
    },
    "Fluconazole": {
        "smiles": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
        "category": "Drug"
    },
    "Dexamethasone": {
        "smiles": "C[C@]12C[C@H](O)[C@@]3(C(=O)CO)C(=O)C=C4[C@@H]([C@@H](O)C[C@]4(C)[C@@]3(F)[C@@H]1CC[C@@H]2O)C",
        "category": "Drug"
    },
    "Penicillin G": {
        "smiles": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@@H]1C(=O)O",
        "category": "Drug"
    },
    "Naproxen": {
        "smiles": "COc1ccc2cc(ccc2c1)[C@@H](C)C(=O)O",
        "category": "Drug"
    },
    "Diclofenac": {
        "smiles": "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
        "category": "Drug"
    },
    "Lidocaine": {
        "smiles": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        "category": "Drug"
    },
    "Vitamin C": {
        "smiles": "OCC(O)C1OC(=O)C(O)=C1O",
        "category": "Vitamin"
    },
    "Vitamin B6": {
        "smiles": "Cc1ncc(CO)c(CO)c1O",
        "category": "Vitamin"
    },
    "Glucose": {
        "smiles": "OCC1OC(O)C(O)C(O)C1O",
        "category": "Biomolecule"
    },
    "Cholesterol": {
        "smiles": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CCC4(C3CCC4O)C)C",
        "category": "Biomolecule"
    },
    "Dopamine": {
        "smiles": "NCCc1ccc(O)c(O)c1",
        "category": "Biomolecule"
    },
    "Serotonin": {
        "smiles": "NCCc1c[nH]c2ccc(O)cc12",
        "category": "Biomolecule"
    },
    "Adenine": {
        "smiles": "Nc1ncnc2[nH]cnc12",
        "category": "Biomolecule"
    },
    "Glycine": {
        "smiles": "NCC(=O)O",
        "category": "Biomolecule"
    },
    "Nicotinamide": {
        "smiles": "NC(=O)c1cccnc1",
        "category": "Biomolecule"
    },
    "Urea": {
        "smiles": "NC(N)=O",
        "category": "Chemical"
    },
    "Benzene": {
        "smiles": "c1ccccc1",
        "category": "Chemical"
    },
    "Curcumin": {
        "smiles": "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
        "category": "Natural Product"
    },
    "Quercetin": {
        "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "category": "Natural Product"
    },
    "Resveratrol": {
        "smiles": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
        "category": "Natural Product"
    },
    "Glutathione": {
        "smiles": "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O",
        "category": "Biomolecule"
    },
    "ATP": {
        "smiles": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
        "category": "Biomolecule"
    }
}

# ---------------- HEADER ----------------

st.title("🧬 MolecuLab Pro")
st.subheader("Advanced Molecular & Drug Property Analyzer")

st.write(
    "Explore molecular structures, calculate chemical descriptors, "
    "and compare different molecules."
)

st.info(
    "Research and educational tool. Calculated properties are "
    "not proof of drug safety or clinical effectiveness."
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🔬 Molecule Explorer")

category = st.sidebar.selectbox(
    "Choose category",
    ["All", "Drug", "Vitamin", "Biomolecule",
     "Chemical", "Natural Product"]
)

search = st.sidebar.text_input("🔍 Search molecule")

available = [
    name for name, data in MOLECULES.items()
    if (category == "All" or data["category"] == category)
    and search.lower() in name.lower()
]

if not available:
    st.warning("No molecules found. Try another search.")
    st.stop()

selected = st.sidebar.selectbox(
    "Select molecule",
    available
)

# ---------------- ANALYSIS FUNCTION ----------------

def analyze_molecule(name, smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    props = {
        "Molecule": name,
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "Exact Mass": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "TPSA (Å²)": round(
            rdMolDescriptors.CalcTPSA(mol), 2
        ),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Total Atoms": mol.GetNumAtoms(),
        "Ring Count": Lipinski.RingCount(mol),
        "Formal Charge": sum(
            atom.GetFormalCharge() for atom in mol.GetAtoms()
        ),
        "Molecular Refractivity": round(
            Descriptors.MolMR(mol), 2
        ),
        "Fraction Csp3": round(
            rdMolDescriptors.CalcFractionCSP3(mol), 3
        ),
        "Canonical SMILES": Chem.MolToSmiles(mol)
    }

    # Simple Lipinski Rule of Five screen
    violations = sum([
        props["Molecular Weight"] > 500,
        props["LogP"] > 5,
        props["H-Bond Donors"] > 5,
        props["H-Bond Acceptors"] > 10
    ])

    props["Lipinski Violations"] = violations
    props["Lipinski Screen"] = (
        "Passes basic screen" if violations == 0
        else f"{violations} rule violation(s)"
    )

    return mol, props

# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs([
    "🧪 Analyze",
    "⚗️ Custom SMILES",
    "📊 Compare Molecules"
])

# ---------------- TAB 1 ----------------

with tab1:
    data = MOLECULES[selected]

    mol, props = analyze_molecule(
        selected, data["smiles"]
    )

    st.header(f"🔬 {selected}")
    st.caption(f"Category: {data['category']}")

    if mol is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("2D Molecular Structure")

            image = Draw.MolToImage(
                mol,
                size=(500, 400),
                kekulize=True
            )

            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Key Properties")

            st.metric(
                "Molecular Weight",
                f"{props['Molecular Weight']} g/mol"
            )

            st.metric(
                "Molecular Formula",
                props["Molecular Formula"]
            )

            st.metric("LogP", props["LogP"])
            st.metric("TPSA", f"{props['TPSA (Å²)']} Å²")

            st.metric(
                "Heavy Atoms",
                props["Heavy Atoms"]
            )

        st.divider()
        st.subheader("📊 Complete Molecular Properties")

        property_df = pd.DataFrame(
            list(props.items()),
            columns=["Property", "Value"]
        )

        st.dataframe(
            property_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("💊 Drug-likeness Screening")
        st.write(props["Lipinski Screen"])

        st.caption(
            "Lipinski screening is a simplified guideline. "
            "Passing it does not establish that a molecule "
            "is a medicine or is safe."
        )

        st.download_button(
            "📥 Download Properties as CSV",
            data=property_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected.replace(' ', '_')}_properties.csv",
            mime="text/csv"
        )

# ---------------- TAB 2 ----------------

with tab2:
    st.header("⚗️ Analyze Your Own Molecule")

    st.write(
        "Paste a valid SMILES string to calculate its "
        "molecular properties."
    )

    custom_name = st.text_input(
        "Molecule name",
        value="Custom Molecule"
    )

    custom_smiles = st.text_area(
        "Enter SMILES",
        placeholder="Example: CC(=O)Oc1ccccc1C(=O)O"
    )

    if st.button("Analyze Custom Molecule"):
        if not custom_smiles.strip():
            st.warning("Please enter a SMILES string.")

        else:
            custom_mol, custom_props = analyze_molecule(
                custom_name.strip() or "Custom Molecule",
                custom_smiles.strip()
            )

            if custom_mol is None:
                st.error(
                    "Invalid SMILES. Please check your input."
                )

            else:
                st.subheader("2D Structure")

                custom_image = Draw.MolToImage(
                    custom_mol,
                    size=(500, 400)
                )

                st.image(
                    custom_image,
                    use_container_width=True
                )

                st.subheader("Molecular Properties")

                st.dataframe(
                    pd.DataFrame(
                        list(custom_props.items()),
                        columns=["Property", "Value"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "📥 Download Custom Molecule CSV",
                    data=pd.DataFrame(
                        list(custom_props.items()),
                        columns=["Property", "Value"]
                    ).to_csv(index=False).encode("utf-8"),
                    file_name="custom_molecule.csv",
                    mime="text/csv"
                )

# ---------------- TAB 3 ----------------

with tab3:
    st.header("📊 Compare Molecules")

    compare_names = st.multiselect(
        "Select 2–5 molecules",
        list(MOLECULES.keys()),
        default=["Aspirin", "Paracetamol"]
    )

    if st.button("Compare Selected Molecules"):
        if len(compare_names) < 2:
            st.warning("Please select at least two molecules.")

        elif len(compare_names) > 5:
            st.warning("Please select no more than five molecules.")

        else:
            comparison = []

            for name in compare_names:
                mol, props = analyze_molecule(
                    name,
                    MOLECULES[name]["smiles"]
                )

                if props:
                    comparison.append(props)

            if comparison:
                comparison_df = pd.DataFrame(comparison)
                comparison_df = comparison_df.set_index("Molecule").T

                st.dataframe(
                    comparison_df,
                    use_container_width=True
                )

                chart_data = pd.DataFrame(comparison)

                st.subheader("Molecular Weight Comparison")

                st.bar_chart(
                    chart_data.set_index("Molecule")[
                        ["Molecular Weight"]
                    ]
                )

                st.download_button(
                    "📥 Download Comparison CSV",
                    data=pd.DataFrame(comparison).to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="molecule_comparison.csv",
                    mime="text/csv"
                )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "MolecuLab Pro | Molecular Chemistry & Computational "
    "Drug Research | Educational Use"
)       