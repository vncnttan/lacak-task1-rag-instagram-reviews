import torch
import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer
from llm.llama_32_1b_instruct_rag import get_llm_response
from ui.ui import init_ui
import streamlit as st

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_embeddings():
    """
    Loads review embeddings from a CSV file and converts them into torch tensors.
    This function performs the following steps:
    1. Reads the embeddings from the CSV file.
    2. Converts the embedding column from string representation to numpy arrays.
    3. Converts the numpy arrays to torch tensors.
    
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The embeddings as a torch tensor.
            - pandas.DataFrame: The DataFrame containing the original embeddings and additional data.
    """
    # When the application first started, load all embedding, embedding models and store them in memory
    # Load embeddings -> To Generate Embeddings, refer to generate-embeddings-csv.ipynb
    print("Loading review embeddings...")
    text_chunks_embeddings_df = pd.read_csv("./embeddings/text_chunks_embeddings-20token.csv")

    # Convert embedding column back to np.array
    text_chunks_embeddings_df["chunk_embedding"] = text_chunks_embeddings_df["chunk_embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))


    # Convert np.array to torch.tensor
    return (
        torch.tensor(np.array(text_chunks_embeddings_df["chunk_embedding"].tolist()), dtype=torch.float32).to(device),
        text_chunks_embeddings_df
    )

def load_embedding_model():
    """
    Loads the Sentence Transformer model 'all-mpnet-base-v2' for generating embeddings.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    # Sentence Transformer Model -> all-mpnet-base-v2 used in generating embeddings
    print("Loading model...")
    embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2',
                                device=device)
    return embedding_model
    

def get_relevant_chunks(prompt: str) -> str:
    """
    Retrieves the top 20 relevant text chunks based on the similarity of their embeddings to the given prompt.
    Args:
        prompt (str): The input prompt for which relevant text chunks are to be found.
    Returns:
        str: A string containing the top 20 relevant text chunks concatenated together, separated by ". ".
    """
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"], st.session_state["text_chunks_embeddings_df"] = load_embeddings()
        st.session_state["embedding_model"] = load_embedding_model()
    
    # Get the embedding of the prompt
    embeddings = st.session_state["embeddings"]
    text_chunks_embeddings_df = st.session_state["text_chunks_embeddings_df"]
    embedding_model = st.session_state["embedding_model"]

    query_embedding = embedding_model.encode(prompt, convert_to_tensor=True)
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]

    # Get top 20 relevant chunks based on the dot scores
    top_20_indices = torch.topk(dot_scores, k=20).indices

    return ". ".join(text_chunks_embeddings_df.iloc[top_20_indices]["sentence_chunk"])

def get_answer(prompt: str) -> str:
    relevant_chunks = get_relevant_chunks(prompt)
    return get_llm_response(prompt, relevant_chunks)

def main():
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"], st.session_state["text_chunks_embeddings_df"] = load_embeddings()
        st.session_state["embedding_model"] = load_embedding_model()

    init_ui(get_answer)

if __name__ == "__main__":
    main()