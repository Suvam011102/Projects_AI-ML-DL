# Part 1: Library Imports and Setting up the Device
# -------------------------------------------------
# Import necessary libraries.
import torch
import os
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Part 2: Loading and Inspecting the PDFs
# ----------------------------------------
# Load PDF files from the specified directory and count the total number of pages.
loader = PyPDFDirectoryLoader("unzipped_files/aircraft_pdfs")
docs = loader.load()
print(f"Total pages across all documents: {len(docs)}")

# Part 3: Setting up Embeddings
# -----------------------------
# Initialize embeddings using a pre-trained model to represent the text data.
embeddings = HuggingFaceInstructEmbeddings(
    model_name="thenlper/gte-small", model_kwargs={"device": DEVICE}
)
# Part 4: Text Splitting
# -----------------------
# Split the loaded documents into smaller chunks to be used for further processing.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)
print(f"Number of text chunks: {len(texts)}")

# Part 5: Creating an Embeddings Database
# ---------------------------------------
# Create an embeddings database using Chroma from the split text chunks.
db = Chroma.from_documents(texts, embedding=embeddings, persist_directory="db")

# Part 6: Setting up LLM (Large Language Model)
# ------------------------------------------------
# Set up the environment variable for Hugging Face and initialize the desired model.
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
# model_id = "tiiuae/falcon-7b-instruct"
# llm_hub = HuggingFaceHub(repo_id=model_id, model_kwargs={"temprature": 0.0, "max_new_tokens": 1200, "max_length": 1200})

# Or use

my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}

params = {
        GenParams.MAX_NEW_TOKENS: 256, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }

LLAMA2_model = Model(
        model_id= 'meta-llama/llama-2-70b-chat', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network", 
        )

llm_hub = WatsonxLLM(model=LLAMA2_model)

# Part 7: Defining the prompt template
# ------------------------------------
# Define the template for the questions that'll be used with the LLM.
template ="""
You are a helpful, respectful, and honest assistant to build aircraft. So, it is important to be precise. The following are the context of knowledge to build aircraft. Based on the context information, answer the follow-up questions.
{context}
Question: {question}
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Part 8: Constructing the Question-Answer API Chain
# --------------------------------------------------
# Build the QA chain, which utilizes the LLM and retriever for answering questions.
api_chain = RetrievalQA.from_chain_type(
    llm=llm_hub,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# Part 9: Querying and Displaying the Results
# -------------------------------------------
output = api_chain("what are the materials for constructing an airplane?")

# Extract and print relevant information from the output.
query = output["query"]
result = output["result"]
print(f"Query: {query}\nResult: {result}\n")

# Print the source documents related to the query.
for idx, doc in enumerate(output["source_documents"], 1):
    print(f"Document {idx}:\n")
    print("Page Content:", doc.page_content)
    print("\nMetadata:")
    for key, value in doc.metadata.items():
        print(f"{key}: {value}")
    print("\n" + "-"*50 + "\n")

    