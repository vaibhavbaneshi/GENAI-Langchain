import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound

# Streamlit Page Setup
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Sidebar: API Key Input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

    # Test API Key Button
    if st.button("Test Groq API"):
        if not groq_api_key.strip():
            st.error("Please provide a valid Groq API Key")
        else:
            try:
                llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)  # ✅ Use a valid model
                result = llm.invoke("Say Hello")
                st.success(result.content)
                st.success('Model Used: ' + result.response_metadata['model_name'])
            except Exception as e:
                st.exception(f"Exception: {e}")

# Input URL
generic_url = st.text_input("URL", label_visibility="collapsed")

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Main Summarize Button
if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video or a website URL")
    else:
        try:
            with st.spinner("Loading and summarizing..."):
                # Initialize LLM
                llm = ChatGroq(model="Gemma2-9b-It", api_key=groq_api_key)

                # Load Documents
                if "youtube.com" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                        docs = loader.load()
                    except (TranscriptsDisabled, NoTranscriptFound):
                        st.error("This YouTube video doesn't have a transcript available. Please try another one.")
                        docs = []
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                if not docs:
                    st.warning("No content could be extracted to summarize.")
                else:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    st.success(output_summary)

        except Exception as e:
            st.exception(f"Loader Error: {e}")