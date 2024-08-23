import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-4"

llm = ChatOpenAI(temperature=0.7, model =llm_model)


embeddings = OpenAIEmbeddings()

loader= PyPDFLoader("./data/simple-NN-module-for-relational-reasoning.pdf")

pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(pages)
print(len(splits))


from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores.chroma import Chroma

persist_directory = "./data/db/chroma"


vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)


# print(vectorstore._collection.count())

query = "Waht do they say about the introduction to neural networks"

docs_resp = vectorstore.similarity_search(query=query, k=3)

print(docs_resp[0].page_content)


vectorstore.persist()