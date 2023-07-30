# Load the necessary libraries
import streamlit as st
from dotenv import load_dotenv
from streamlit.components import HTML, Button, Text
from streamlit.directives import download

# Load the environment variables
load_dotenv()

# Set up the Streamlit app
st.set_page_config(page_title="Ask Manyu", page_icon=":books:")

# Define the functions to get the text chunks and vector store
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)  # return as a list
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    # Use Instructor Embedding model
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vector_store

# Define the function to get the conversation chain
def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Define the handler for user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content
            ), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content
            ), unsafe_allow_html=True)

# Define the main function
def main():
    # Load the environment variables
    load_dotenv()

    # Set up the Streamlit app
    st.set_page_config(page_title="Ask Manyu", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Upload the PDF file
    pdf_file = st.file_uploader("Choose a PDF file", accept_ Multiple=True)
    # Upload the PDF file
    pdf_file = st.file_uploader("Choose a PDF file", accept_multiple=True)

    # Extract the text from the uploaded PDF file
    text = extract_text_from_pdf(pdf_file)

    # Split the text into chunks
    text_chunks = split_text_into_chunks(text, chunk_size=1000)

    # Create the vector store
    vector_store = get_vector_store(text_chunks)

    # Generate the conversation chain
    conversation_chain = get_conversation_chain(vector_store)

    # Display the conversation chain
    st.write(conversation_chain, unsafe_allow_html=True)

main()
