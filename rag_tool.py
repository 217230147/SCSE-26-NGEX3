### COMPLETE THE CODE  ###

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
from rag import retrieve_policy

def search_policy_tool(query: str) -> str:
    """
    Call vector store to retrieve university IT policies.
    Returns a concatenated string of retrieved document contents.
    """
    docs = retrieve_policy(query, k=2)
    if not docs:
        return "No relevant documents found."
    context = "\n\n".join([doc.page_content for doc in docs])
    return context