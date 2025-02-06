# import argparse
# from langchain_chroma import Chroma  # Updated import
# from langchain.prompts import ChatPromptTemplate
# from langchain_ollama import OllamaLLM  

# from get_embedding_function import get_embedding_function

# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """


# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)


# def query_rag(query_text: str):
#     # Prepare the DB.
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB.
#     results = db.similarity_search_with_score(query_text, k=5)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     # print(prompt)

#     model = OllamaLLM(model="deepseek-r1:8b")
#     # model = Ollama (model= "supachai/llama-3-typhoon-v1.5")
#     response_text = model.invoke(prompt)

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
#     return response_text


# if __name__ == "__main__":
#     main()


# import argparse
# from langchain_chroma import Chroma  # Correct import for Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_ollama import OllamaLLM, OllamaEmbeddings  # Correct imports for OllamaLLM and OllamaEmbeddings

# from get_embedding_function import get_embedding_function

# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# "ตอบคำถามโดยอ้างอิงจากบริบทต่อไปนี้เท่านั้น":

# {context}

# ---

# ตอบคำถามโดยอ้างอิงจากบริบทข้างต้น: {question}
# """

# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)

# def query_rag(query_text: str):
#     # Prepare the DB.
#     embedding_function = get_embedding_function()  # Assuming this is using OllamaEmbeddings correctly
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB.
#     results = db.similarity_search_with_score(query_text, k=5)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     # Use OllamaLLM for querying
#     # model = OllamaLLM(model="supachai/llama-3-typhoon-v1.5")  
#     model = OllamaLLM(model="cageyv/typhoon-7b") 
#     # model = OllamaLLM(model="deepseek-r1:8b") # Correct usage of OllamaLLM
#     response_text = model.invoke(prompt)

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
#     return response_text

# if __name__ == "__main__":
#     main()






import argparse
from langchain_chroma import Chroma  # Updated import
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
ตอบคำถามโดยอ้างอิงจากบริบทต่อไปนี้เท่านั้น:

{context}

---

ตอบคำถามเป็นภาษาไทยโดยอ้างอิงจากบริบทข้างต้น: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # model = OllamaLLM(model="deepseek-r1:8b")
    model = OllamaLLM(model="supachai/llama-3-typhoon-v1.5")
    # model = OllamaLLM(model="cageyv/typhoon-7b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"คำตอบ: {response_text}\nแหล่งที่มา: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
