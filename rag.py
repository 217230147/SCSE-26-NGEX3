### COMPLETE THE CODE ###

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from policy_loader import load_policy_documents

documents = load_policy_documents()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="university_support",
    embedding_function=embeddings
)

vector_store.add_documents(documents)

def retrieve_policy(question, k=2):
    """Retrieve relevant documents based on the question."""
    results = vector_store.similarity_search(question, k=k)
    return results