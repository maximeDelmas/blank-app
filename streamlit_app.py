import os
import json
import dspy
import pandas as pd
import streamlit as st
from app.call_llm import ClimateChangeDriverInfer

st.set_page_config(
    layout="centered",
    page_title="Basinghall Explorer",
    page_icon="🌎",
    initial_sidebar_state="expanded",
)

c1, c2 = st.columns([1, 1])

st.image(
    "images/basinghall.png",
    width=300,
)
st.title("Climate Impact Explorer")

st.markdown("Welcome to the Basinghall Climate Impact Explorer! This tool helps you explore climate impact drivers and their effects on various sectors. Use the sidebar to customize your search and view detailed insights.")


# List of NACE categories
NACE_CATEGORIES = ["Construction of buildings", "Production of electricity", "Manufacture of textiles"]
COLLECTION_NAMES = ["IPCC_WG_REPORTS", "IPCC_SUPPLEMENTARIES", "NGFS_2024_REPORTS"]


print("Read data ...")
nace = pd.read_csv("data/categories/KeyRisk-NACE.tsv", sep="\t")
nace_categories = nace["Description"].unique()
nace = nace.drop_duplicates()

cid = pd.read_csv(
    "data/categories/KeyRisk-CID.tsv", sep="\t"
)
cid_categories = cid["CID Category"].unique()

# For the location
cid_change = pd.read_csv(
    "data/categories/KeyRisk-CID-changes.tsv",
    sep="\t",
)
regions = cid_change["Region"].unique()
print("Ok !")

# Load the retriever data
results_dir = "data/results"
retriever_data = {}

for filename in os.listdir(results_dir):
    if filename.endswith(".json"):
        with open(os.path.join(results_dir, filename), "r") as file:
            retriever_data[os.path.splitext(filename)[0]] = json.load(file)

# Init the inference model
inference = ClimateChangeDriverInfer()

# Sidebar
selected_cid_category = st.sidebar.selectbox(
    "Select a Climate Impact Driver", cid_categories
)
selected_data = cid[cid["CID Category"] == selected_cid_category]
cid_title = selected_data["CID Category"].values[0]
cid_description = selected_data["Brief Description"].values[0]
cid_code = selected_data["CID code"].values[0]

st.session_state.disabled = False

selected_region = st.sidebar.selectbox("Select a Localisation", regions)
change = cid_change[(cid_change["CID code"] == cid_code) & (cid_change["Region"] == selected_region)]["Direction"].values[0]
if change == "NR":
    st.session_state.disabled = True
else:
    st.session_state.disabled = False

change_map = {
"HCI": "High confidence of increase", 
"MCI": "Medium confidence of increase", 
"HCD": "High confidence of decrease", 
"MCD": "Medium confidence of decrease",
"LCDC": "Low confidence of direct changes",
"NR": "Not relevant"
}

selected_nace = st.sidebar.selectbox("Select a NACE category", NACE_CATEGORIES)

selected_scenario = st.sidebar.selectbox("Select a scenario", ["SSP1-1.9", "SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"])

selected_timeline = st.sidebar.selectbox("Select a timeline", ["20 years", "50 years", "100 years"])

seleced_model = st.sidebar.selectbox("Select a model", ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo-0125"])
api_key =  st.sidebar.text_input("Insert your Api-key", type="password")
model = f"openai/{seleced_model}"

llm = dspy.LM(model, api_key=api_key)
# llm = dspy.LM("openai/gpt-4-turbo", api_key=api_key)
dspy.configure(lm=llm)
print(llm.model)

# check the api key
if api_key == "":
    st.session_state.disabled = True
else:
    st.session_state.disabled = False


# Initialize session state for checkboxes if not already done
if "selected_collections" not in st.session_state:
    st.session_state.selected_collections = [True] * len(COLLECTION_NAMES)

# Function to select all checkboxes
def select_all():
    st.session_state.selected_collections = [True] * len(COLLECTION_NAMES)

# Function to unselect all checkboxes
def unselect_all():
    st.session_state.selected_collections = [False] * len(COLLECTION_NAMES)

# Create a collapsible sub-panel in the sidebar
with st.sidebar.expander("Select Collections"):
    # Add buttons to select/unselect all
    if st.button("Select All"):
        select_all()
    if st.button("Unselect All"):
        unselect_all()

    # Add checkboxes for each collection name
    selected_indexes = []
    for i, name in enumerate(COLLECTION_NAMES):
        if st.checkbox(name, value=st.session_state.selected_collections[i]):
            selected_indexes.append(i)
            st.session_state.selected_collections[i] = True
        else:
            st.session_state.selected_collections[i] = False

# Settings of other filters

k_reranker = st.sidebar.number_input("k_reranker", min_value=1, max_value=50, value=5)
sentiment_threshold = st.sidebar.slider(
    "sentiment_threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.1
)
reranker_threshold = st.sidebar.slider(
    "reranker_threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.1
)

# Initialize the results to None
prediction_opportunities = None
prediction_risks = None

if st.sidebar.button("Submit", disabled=st.session_state.disabled):
    print(selected_indexes)
    
    # the cid text
    cid_text = f"{cid_title} ({cid_description})" if cid_description else cid_title
    
    progress_bar = st.progress(0)
    progress_bar.progress(50)
    retrieval = retriever_data[selected_nace][cid_title]
    
    # filter in the collection
    filtered_retrieval = [item for item in retrieval if item["metadatas"]["collection_idx"] in selected_indexes]
    
    # filter on the sentiment threshold
    filtered_retrieval = [item for item in filtered_retrieval if item["classification"]["score"] >= sentiment_threshold]
    
    # filter on the reranker threshold
    filtered_retrieval = [item for item in filtered_retrieval if item["reranker"]["score"] >= reranker_threshold]
    
    # split orrortunities and risks
    opportunities = [item for item in filtered_retrieval if item["classification"]["label"] == "opportunity"]
    risks = [item for item in filtered_retrieval if item["classification"]["label"] == "risk"]
    
    # sort by reranker score
    opportunities = sorted(opportunities, key=lambda x: x["reranker"]["score"], reverse=True)
    risks = sorted(risks, key=lambda x: x["reranker"]["score"], reverse=True)
    
    # get the top k_reranker
    opportunities = opportunities[:k_reranker]
    risks = risks[:k_reranker]
    
    # get the text
    opportunities_text = [item["display_text"] for item in opportunities]
    risks_text = [item["display_text"] for item in risks]
    
    prediction_opportunities, prediction_risks = inference(risks, opportunities, cid_text, selected_nace)
    progress_bar.progress(100)
    print(llm.inspect_history(2))


st.subheader("Location info", divider=True)

change = change_map[change]
st.text(f"According to IPCC AR56 reports and predictions, there is a {change} of {cid_title} in {selected_region}.")

st.header("Risks", divider=True)
if prediction_risks is not None:
    # Display prediction_risks
    st.write(f"Reasoning: {prediction_risks.reasoning}")
    st.write(f"Answer: {prediction_risks.answer}")
    # Add the references
    for i, risk in enumerate(risks):
        risk_expander = st.expander(f"Evidence {i+1}")
        with risk_expander:
            st.markdown(f"**Document:** {risk['metadatas']['doc_id']}")
            sections = [
                ">" * i + " " + risk["metadatas"][f"Header {i}"]
                for i in range(6)
                if f"Header {i}" in risk["metadatas"]
            ]
            text_sections = "\n".join(sections)
            st.markdown(text_sections)
            st.markdown(f"Content: *{risk['display_text']}*")

st.header("Opportunities", divider=True)

if prediction_opportunities is not None:
    # Display prediction_opportunities
    st.write(f"Reasoning: {prediction_opportunities.reasoning}")
    st.write(f"Answer: {prediction_opportunities.answer}")

    # Add the references
    for i, opportunity in enumerate(opportunities):
        with st.expander(f"Evidence {i+1}", expanded=True):
            st.markdown(f"**Document:** {opportunity['metadatas']['doc_id']}")
            sections = [
                ">" * i + " " + opportunity["metadatas"][f"Header {i}"]
                for i in range(6)
                if f"Header {i}" in opportunity["metadatas"]
            ]
            text_sections = "\n".join(sections)
            st.markdown(text_sections)
            st.markdown(f"Content: *{opportunity['display_text']}*")
