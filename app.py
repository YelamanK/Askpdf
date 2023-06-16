from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pinecone

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    user_question = st.text_input("Ask a question about your PDF:")

# extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      
      # create a Pinecone text vector index
      pinecone.create_index(index_name='pdf-text-index', dimension=512, metric='cosine')
      
      chunks = []
      # upload your text chunks to the index
      for chunk in chunks:
        pinecone.index(index_name='pdf-text-index', data=[chunk])

      # perform a similarity search on the index
      results = pinecone.query(index_name='pdf-text-index',                     query=user_question, top_k=50)
      # get the text chunks corresponding to the results
      chunks = [result['data'] for result in results]
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
     
      if user_question:
        docs = knowledge_base.similarity_search(user_question, top_k=50)
        
        llm = OpenAI(max_length=1000)
        chain = load_qa_chain(llm, chain_type="stuff")
        
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question, max_outputs=3)
          print(cb)
           
        st.write(response)
    

if __name__ == '__main__':
    main()