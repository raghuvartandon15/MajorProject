import ssl
import os
import validators
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from pytube import YouTube

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Summarize Text From YT or Website", page_icon="🦜")
st.title("Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Ensure API key is loaded correctly
if not groq_api_key:
    st.error("API Key not found! Please set it in the .env file.")
else:
    # Input field for URL
    generic_url = st.text_input("Enter a YouTube or Website URL")

    # LangChain Model using Groq API
    llm = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=groq_api_key)

    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content:{text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    if st.button("Summarize the Content"):
        # Validate inputs
        if not generic_url.strip():
            st.error("Please provide a valid URL.")
        elif not validators.url(generic_url):
            st.error("Invalid URL format.")
        else:
            try:
                with st.spinner("Processing..."):
                    # Load content based on URL type
                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        try:
                            yt = YouTube(generic_url)
                            if not yt.captions:
                                st.error("No captions available for this video.")
                                st.stop()
                            video_text = yt.captions.get_by_language_code("en").generate_srt_captions()
                            docs = [{"text": video_text}]
                        except Exception as yt_error:
                            st.error(f"Error processing YouTube video: {yt_error}")
                            st.stop()
                    else:
                        # Load content for a website
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                            },
                        )
                        docs = loader.load()

                    # Chain for summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success(output_summary)
            except Exception as e:
                st.exception(f"Exception: {e}")
