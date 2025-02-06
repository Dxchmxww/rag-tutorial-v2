from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# # get_embedding_function.py

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load the model and tokenizer
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained("sambanovasystems/SambaLingo-Thai-Chat", use_fast=False)
#     model = AutoModelForCausalLM.from_pretrained("sambanovasystems/SambaLingo-Thai-Chat", torch_dtype=torch.float16)
#     model.eval()  # Set to evaluation mode for embeddings
#     return tokenizer, model

# # Create an embedding class with embed_query method
# class EmbeddingFunction:
#     def __init__(self):
#         # Load the model and tokenizer once when the class is initialized
#         self.tokenizer, self.model = load_model()

#     def embed_query(self, query: str):
#         """Generate embedding for a given query."""
#         inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
        
#         # Extract embeddings from the last hidden state (using mean across tokens)
#         embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
#         return embeddings

# # Create the function that returns the embedding instance
# def get_embedding_function():
#     return EmbeddingFunction()


# Now pass `embedding_function` to Chroma


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    
    
    # embedding = embeddings.embed_query("Sample text for dimensionality check.")
    # print(len(embedding)) 
    # embeddings = OllamaEmbeddings(model="cageyv/typhoon-7b")
    embeddings = OllamaEmbeddings(model="supachai/llama-3-typhoon-v1.5")
    # embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    return embeddings
