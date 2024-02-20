import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os




load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="GOOGLE_API_KEY")

# Set up the model
generation_config = {
  "temperature": 0.4,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def get_transcript_from_url(url):
    # get the transcript in youtube video
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info = True,
    )
    transcript = loader.load()
    return transcript

def get_vectore_store_from_transcript(transcript):
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits = text_splitter.split_documents(transcript)
    google_embeddings = GoogleGenerativeAIEmbeddings( model = "models/embedding-001")
    vectorestore = Chroma.from_documents(documents=splits, embedding = google_embeddings) 
    return vectorestore

def get_retriever_chain():
    # prompt 
    prompt = ChatPromptTemplate.from_template(
        """As an Youtube expert who has in depth knowledge on the transcript of a specific video, answer the following question based only the provided context:
        <context>
        {context}
        </context>

        Question: {input}""")
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    # document chain 
    document_chain = create_stuff_documents_chain(llm, prompt)

    return document_chain

def get_retrieval_chain(vector_store, document_chain):
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
    

def main():
    st.set_page_config(page_title="Chat with YT", page_icon="▶️")
    st.title("Chat With YT")
    st.header("We are here to satisfy your inquiries")
    
    # Sidebar
    with st.sidebar:
        st.header("Input YouTube URL")
        youtube_url = st.text_input("Enter YouTube URL")
    
    if youtube_url is None or youtube_url == "":
        st.info("Please enter a Youtube video's URL!")
    
    else:
        # Session State:
        if "chat-history" not in st.session_state:
            st.session_state.chat_history = []

        # User query
        user_query = st.chat_input("Say something") 
        
        # Transcript
        transcript = get_transcript_from_url(youtube_url)
        st.subheader(f"Video: {transcript[0].metadata['title']}")

        # vectore store
        vector_store = get_vectore_store_from_transcript(transcript)
        
        #retriever chain
        retriever = get_retriever_chain()

        #retrieval chain
        retrieval_chain = get_retrieval_chain(vector_store, retriever)

        #response
        if user_query is not None and user_query!="":
          with st.chat_message("user"):
              st.write(user_query)
          response = retrieval_chain.invoke({
              "input": user_query,
          })
          with st.chat_message("ai"):
            st.write(response["answer"])
            
if __name__ == "__main__":
    main()