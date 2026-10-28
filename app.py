import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Lipinski,
    rdMolDescriptors,
    AllChem,
    Draw
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Molecular Analysis Lab",
    page_icon="🧬",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    padding: 10px;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #64748b;
    margin-bottom: 25px;
}
div[data-testid="stMetric"] {
    background: #f0f7ff;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🧬 Molecular Analysis Lab</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore molecular structures, properties and 3D models'
    '</div>',
    unsafe_allow_html=True
)

# ---------------- MOLECULE LIBRARY ----------------

MOLECULES = {
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",

    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",

    "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",

    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",

    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",

    "Cholesterol":
        "CC(C)CCCC(C)C1CCC2C1(CCC3C2CCC4"
        "(C3(CCC4O)C)C)C",

    "Quercetin":
        "O=c1c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c2c1O",

    "Nicotinamide": "NC(=O)c1cccnc1",

    "Benzene": "c1ccccc1",

    "Vitamin C": "O=C1OC(C(O)CO)C(O)=C1O"
}

# ---------------- PROPERTY CALCULATOR ----------------

def analyze_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles.strip())

    if mol is None:
        return None, None

    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    exact_mass = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable = Lipinski.NumRotatableBonds(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = (
        rdMolDescriptors.CalcNumAromaticRings(mol)
    )
    fraction_csp3 = Descriptors.FractionCSP3(mol)
    formal_charge = Chem.GetFormalCharge(mol)

    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10
    ])

    data = {
        "Molecular Formula": formula,
        "Molecular Weight (g/mol)": round(mw, 3),
        "Exact Mass (Da)": round(exact_mass, 4),
        "LogP": round(logp, 3),
        "TPSA (Å²)": round(tpsa, 3),
        "H-Bond Donors": hbd,
        "H-Bond Acceptors": hba,
        "Rotatable Bonds": rotatable,
        "Heavy Atoms": heavy_atoms,
        "Total Rings": rings,
        "Aromatic Rings": aromatic_rings,
        "Fraction Csp3": round(fraction_csp3, 3),
        "Formal Charge": formal_charge,
        "Lipinski Violations": violations,
        "Canonical SMILES": Chem.MolToSmiles(mol)
    }

    return mol, data


# ---------------- SIDEBAR ----------------

st.sidebar.title("🔬 Navigation")

st.sidebar.info(
    "A cheminformatics tool for educational "
    "molecular analysis."
)

st.sidebar.caption(
    "Properties are calculated using RDKit."
)

# ---------------- TABS ----------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Analyze",
    "📚 Molecule Library",
    "⚖️ Compare",
    "ℹ️ About"
])

# ---------------- ANALYZE ----------------

with tab1:

    st.subheader("Analyze a Molecule")

    selected = st.selectbox(
        "Choose an example or select Custom",
        ["Custom"] + list(MOLECULES.keys())
    )

    if selected == "Custom":
        default_smiles = ""
    else:
        default_smiles = MOLECULES[selected]

    smiles = st.text_area(
        "Enter SMILES",
        value=default_smiles,
        height=90,
        placeholder="Paste a valid SMILES string..."
    )

    if st.button(
        "🔍 Analyze Molecule",
        type="primary",
        use_container_width=True
    ):

        if not smiles.strip():
            st.warning("Please enter a SMILES string.")

        else:

            mol, data = analyze_smiles(smiles)

            if mol is None:
                st.error(
                    "Invalid SMILES. Please check your input."
                )

            else:

                st.session_state["mol_data"] = data
                st.session_state["mol_smiles"] = smiles

                st.success("Molecule analyzed successfully!")

    if "mol_data" in st.session_state:

        data = st.session_state["mol_data"]
        smiles_used = st.session_state["mol_smiles"]

        mol, _ = analyze_smiles(smiles_used)

        st.divider()
        st.subheader("📐 2D Molecular Structure")

        st.image(
            Draw.MolToImage(mol, size=(650, 400)),
            use_container_width=True
        )

        st.subheader("📊 Key Properties")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Molecular Weight",
            f'{data["Molecular Weight (g/mol)"]:.2f}'
        )

        c2.metric(
            "LogP",
            f'{data["LogP"]:.2f}'
        )

        c3.metric(
            "TPSA",
            f'{data["TPSA (Å²)"]:.2f} Å²'
        )

        c4, c5, c6 = st.columns(3)

        c4.metric("H-Bond Donors", data["H-Bond Donors"])
        c5.metric("H-Bond Acceptors", data["H-Bond Acceptors"])
        c6.metric("Rotatable Bonds", data["Rotatable Bonds"])

        st.subheader("🧬 Complete Property Report")

        property_df = pd.DataFrame({
            "Property": list(data.keys()),
            "Value": list(data.values())
        })

        st.dataframe(
            property_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("💊 Lipinski Rule of Five")

        st.write(
            "Number of rule limits exceeded:",
            data["Lipinski Violations"]
        )

        if data["Lipinski Violations"] == 0:
            st.success(
                "No Lipinski Rule of Five violations."
            )
        else:
            st.warning(
                "One or more Lipinski limits are exceeded."
            )

        st.caption(
            "This screening rule does not establish "
            "drug safety or effectiveness."
        )

        st.download_button(
            "⬇️ Download Full Report (CSV)",
            data=property_df.to_csv(index=False).encode("utf-8"),
            file_name="molecular_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ---------------- 3D STRUCTURE ----------------

        st.divider()
        st.subheader("🌐 3D Molecular Structure")

        if st.button("Generate 3D Model"):

            if mol.GetNumHeavyAtoms() > 150:
                st.warning(
                    "3D generation is limited to molecules "
                    "with at most 150 heavy atoms in this demo."
                )

            else:

                mol3d = Chem.AddHs(mol)

                params = AllChem.ETKDGv3()
                params.randomSeed = 42

                result = AllChem.EmbedMolecule(
                    mol3d,
                    params
                )

                if result != 0:
                    st.error(
                        "3D coordinates could not be generated."
                    )

                else:

                    AllChem.MMFFOptimizeMolecule(mol3d)

                    conf = mol3d.GetConformer()

                    xyz = [
                        conf.GetAtomPosition(i)
                        for i in range(mol3d.GetNumAtoms())
                    ]

                    fig = go.Figure()

                    # Draw bonds
                    for bond in mol3d.GetBonds():

                        a = bond.GetBeginAtomIdx()
                        b = bond.GetEndAtomIdx()

                        fig.add_trace(go.Scatter3d(
                            x=[xyz[a].x, xyz[b].x, None],
                            y=[xyz[a].y, xyz[b].y, None],
                            z=[xyz[a].z, xyz[b].z, None],
                            mode="lines",
                            line=dict(width=5),
                            hoverinfo="skip",
                            showlegend=False
                        ))

                    # Draw atoms
                    fig.add_trace(go.Scatter3d(
                        x=[p.x for p in xyz],
                        y=[p.y for p in xyz],
                        z=[p.z for p in xyz],
                        mode="markers+text",
                        text=[
                            atom.GetSymbol()
                            for atom in mol3d.GetAtoms()
                        ],
                        textposition="top center",
                        marker=dict(
                            size=6,
                            color=[
                                atom.GetAtomicNum()
                                for atom in mol3d.GetAtoms()
                            ],
                            colorscale="Viridis"
                        ),
                        hovertext=[
                            f"Atom {i}: "
                            f"{mol3d.GetAtomWithIdx(i).GetSymbol()}"
                            for i in range(mol3d.GetNumAtoms())
                        ],
                        hoverinfo="text",
                        showlegend=False
                    ))

                    fig.update_layout(
                        height=500,
                        scene=dict(
                            xaxis_title="X",
                            yaxis_title="Y",
                            zaxis_title="Z",
                            aspectmode="data"
                        ),
                        margin=dict(l=0, r=0, t=20, b=0)
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.caption(
                        "Computed 3D conformer; not an "
                        "experimentally determined structure."
                    )


# ---------------- MOLECULE LIBRARY ----------------

with tab2:

    st.subheader("📚 Molecule Library")

    st.write(
        "Choose a molecule to view its basic properties."
    )

    library_name = st.selectbox(
        "Select a molecule",
        list(MOLECULES.keys()),
        key="library_choice"
    )

    if st.button("View Molecule Properties"):

        lib_mol, lib_data = analyze_smiles(
            MOLECULES[library_name]
        )

        if lib_mol is not None:

            st.subheader(library_name)

            st.image(
                Draw.MolToImage(lib_mol, size=(500, 350)),
                use_container_width=True
            )

            st.write(
                "Molecular Formula:",
                lib_data["Molecular Formula"]
            )

            st.write(
                "Molecular Weight:",
                lib_data["Molecular Weight (g/mol)"],
                "g/mol"
            )

            st.write(
                "Exact Mass:",
                lib_data["Exact Mass (Da)"],
                "Da"
            )

            st.write(
                "LogP:",
                lib_data["LogP"]
            )

            st.write(
                "TPSA:",
                lib_data["TPSA (Å²)"],
                "Å²"
            )

            st.code(MOLECULES[library_name])


# ---------------- COMPARE MOLECULES ----------------

with tab3:

    st.subheader("⚖️ Compare Two Molecules")

    option_names = list(MOLECULES.keys())

    left, right = st.columns(2)

    with left:

        mol_a_name = st.selectbox(
            "First molecule",
            option_names,
            index=0
        )

    with right:

        mol_b_name = st.selectbox(
            "Second molecule",
            option_names,
            index=1
        )

    if st.button(
        "Compare Molecules",
        type="primary"
    ):

        _, data_a = analyze_smiles(
            MOLECULES[mol_a_name]
        )

        _, data_b = analyze_smiles(
            MOLECULES[mol_b_name]
        )

        compare_keys = [
            "Molecular Formula",
            "Molecular Weight (g/mol)",
            "Exact Mass (Da)",
            "LogP",
            "TPSA (Å²)",
            "H-Bond Donors",
            "H-Bond Acceptors",
            "Rotatable Bonds",
            "Heavy Atoms",
            "Total Rings",
            "Aromatic Rings",
            "Lipinski Violations"
        ]

        comparison = pd.DataFrame({
            "Property": compare_keys,
            mol_a_name: [data_a[k] for k in compare_keys],
            mol_b_name: [data_b[k] for k in compare_keys]
        })

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Comparison CSV",
            comparison.to_csv(index=False).encode("utf-8"),
            "molecule_comparison.csv",
            "text/csv"
        )


# ---------------- ABOUT ----------------

with tab4:

    st.subheader("About This Project")

    st.write("""
    Molecular Analysis Lab is an educational
    cheminformatics application built with Python,
    Streamlit and RDKit.

    Features include:

    - Molecular structure visualization
    - Chemical descriptor calculations
    - Lipinski Rule of Five screening
    - 3D conformer generation
    - Molecule library
    - Molecular property comparison
    - CSV report downloads
    """)

    st.info(
        "This application analyzes valid molecular SMILES. "
        "Protein and DNA sequence analysis requires "
        "additional specialized tools."
    )

    st.caption(
        "Designed for educational and research exploration."
    )

st.divider()

st.caption(
    "Molecular Analysis Lab | Python • Streamlit • RDKit"
)
