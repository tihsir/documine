

# Importing libraries
from bs4 import BeautifulSoup
import urllib
from urllib import request
import urllib.request as ur
import requests
import wikipedia
import random
from constants import *
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import random
import chainlit as cl
from constants import *
import os
import asyncio
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from chainlit.callback import AsyncLangchainCallbackHandler


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
import asyncio

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
from langchain_core.documents import Document
from langchain.llms import OpenAI
import os
from langchain.embeddings import OpenAIEmbeddings

# from huggingface_hub import hf_hub_download

import pandas as pd
import os
from constants import *
from dotenv import load_dotenv
from langchain_core.documents import Document

# Loads environment variables from .env
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Include your open api key in .env")

def runChainlit():
        cl.run(
            host="0.0.0.0",
            port=80,
            debug=True,
        )
        
# ========================== Scraper ==================================

all_titles = []

# scrapes all the relevant titles.
def scrapeLinks(url, depth, num_links_per_page=10):

    # url filtering to remove wiki assets:
    if (len(url.split("/")) != 5):
        return 
    last_path = url.split("/")[4]
    if last_path.startswith("Wikipedia:") \
        or last_path.startswith("Category:")\
        or last_path.startswith("Template:") \
        or last_path.startswith("File:") \
        or last_path.startswith("Help:") \
        or last_path.startswith("Special:"):
        return

    global all_titles
    all_titles.append(url)
    
    # end condition
    if (depth == 0):
        return
    
    response = requests.get(
        url=url,
    )

    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find(id="firstHeading")
    allLinks = soup.find(id="bodyContent").find_all("a")
    random.shuffle(allLinks)

    # limit to top n links
    allLinks = allLinks[:num_links_per_page]
    
    # Recursively scrape links
    for link in allLinks:
        try: 
            if link['href'].find("/wiki/") == -1:
                continue
                
            # Use this link to scrape
            scrapeLinks("https://en.wikipedia.org" + link['href'], depth - 1)
        except:
            pass

def scrapeWikiLinksToCSV(search_term):
    
    global all_titles
    
    # Search for similar wiki titles and try all relevant routes:
    relevant_routes = wikipedia.search(search_term)
    for i in relevant_routes:
        try:
            # construct url:
            start_url = wikipedia.page(i).url
            
            # scrapelinks and accumulate in global all_titles
            scrapeLinks(start_url, SEARCH_DEPTH, NUM_LINKS_PER_PAGE)
        except:
            pass
    
    df = pd.DataFrame(columns=["url", "title", "content"])
    wikipedia.set_lang("en")
    for link in all_titles:
        # get basepath:
        title = link.split('/')[-1]
        try:

            # only get the first NUM_SENTENCES_FROM_WIKI amount of sentences
            content = wikipedia.summary(title, sentences=NUM_SENTENCES_FROM_WIKI)
                
            # preprocess content to remove "==== xxx ===="
            content = re.sub(r'==.*?==', '', content)  
            
            # add to dataframe
            df.loc[len(df)] = [link, title, content]
        except:
            pass
    
    # save as csv.
    df.to_csv('scraped.csv', index=False)
    return df

# ========================== Chroma ==================================
    
def createChroma(search_term):
    # Wikipedia DF
    wiki_df = scrapeWikiLinksToCSV(search_term)
    texts_from_wiki = wiki_df['content'].tolist()  
    documents_from_wiki = [ Document(
        page_content= x,
        metadata={}
    ) for x in texts_from_wiki]

    # User DF
    
    # Collate and gather text data from the PDFs in DATA_DIR
    loader = PyPDFDirectoryLoader(DATA_DIR)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    docs = loader.load()
    documents_from_user = text_splitter.split_documents(docs)
    
    # define openAI embeddings and LLMs
    llm = OpenAI()
    embeddings = OpenAIEmbeddings()

    # Consolidate data:
    all_documents = documents_from_wiki + documents_from_user
    
    # create ChromaDB from documents.
    print("Creating Chroma Database from documents... This may take a while.")
    chroma_db = Chroma.from_documents(documents=all_documents, embedding=embeddings, persist_directory="./chroma_db")
    
    return chroma_db


        
def startFlask():
    
     # Start Flask Server:
    app = Flask(__name__)
    
    # Set the upload folder and allowed extensions
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'pptx', 'csv'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Ensure the upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


    # Flask endpoint for Chrome extension
    @app.route('/chat', methods=['POST'])
    def chat():
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "Message is required"}), 400

        # Use asyncio to call the Chainlit message handler
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            chain = conversation_chain
            res = loop.run_until_complete(chain.acall(user_input))
            answer = res["result"]
            return jsonify({"response": answer})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            loop.close()
    
    
def startChainLit():
    
    # chainlit portion:
    # Get prompt template for chain
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)


    # init chain
    conversation_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
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
    
    # Start Chainlit in a separate thread
    import threading
    chainlit_thread = threading.Thread(target=runChainlit, daemon=True)
    chainlit_thread.start()

    # Start Flask server
    app.run(host="0.0.0.0", port=PORT_NUM, debug=True)

def main(search_term="barack"):

    chroma_db = createChroma(search_term)
    startChainLit()
    startFlask()
    

if __name__ == "__main__": 
    main()