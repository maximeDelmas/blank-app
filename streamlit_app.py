import pandas as pd
import streamlit as st


st.set_page_config(
    layout="centered",
    page_title="Basinghall Explorer",
    page_icon="🌎",
    initial_sidebar_state="expanded",
)

print("Read data ...")
nace = pd.read_csv("data/categories/KeyRisk-NACE.tsv", sep="\t")
nace_categories = nace["Description"].unique()
nace = nace.drop_duplicates()

cid = pd.read_csv(
    "data/categories/KeyRisk-CID.tsv", sep="\t"
)
cid_categories = cid["CID Category"].unique()
print("Ok !")

selected_cid_category = st.sidebar.selectbox(
    "Select a Climate Impact Driver", cid_categories
)
selected_data = cid[cid["CID Category"] == selected_cid_category]
cid_title = selected_data["CID Category"].values[0]
cid_description = selected_data["Brief Description"].values[0]
cid_code = selected_data["CID code"].values[0]

print(f"Selected CID: {cid_title}")
print(f"Description: {cid_description}")
print(f"Code: {cid_code}")
