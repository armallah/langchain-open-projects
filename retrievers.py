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
    chunk_size = 1000,
    chunk_overlap = 200
)

splits = text_splitter.split_documents(pages)
# print(len(splits))
# 

from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores.chroma import Chroma

persist_directory = "./data/db/chroma"


vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)


vectorstore.persist()

#laod
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k":2})
docs= retriever.get_relevant_documents("tell me about simple NNs")
# print(retriever.search_type)
# print(docs[0].page_content)

# make a chain to answer questiosn
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def process_llm_response(llm_response):
    print(llm_response['result'])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

query = "tell me more about relational reasoning"

llm_response = qa_chain(query)

print(process_llm_response(llm_response))