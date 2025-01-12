import chainlit as cl
from constants import *

from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA 
from langchain.callbacks import StdOutCallbackHandler
from argparse import ArgumentParser
import json
import os
import chainlit as cl
from huggingface_hub import hf_hub_download
from langchain.llms import OpenAI
import os
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
from constants import *

# Use OpenAI() embeddings
llm = OpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
db.get()


# Get prompt template for chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# init chain
conversation_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
        callbacks=[StdOutCallbackHandler()]
    )   

# chainlit async
@cl.on_chat_start
async def start():
    chain = conversation_chain
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, What would you like to know about the topic?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]

    await cl.Message(content=answer).send()