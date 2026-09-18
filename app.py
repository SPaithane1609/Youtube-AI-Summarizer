import re
import json
import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError
from youtube_transcript_api import YouTubeTranscriptApi

# =========================================================
# ⚙️ CONFIGURATION - HARDCODE YOUR ENDPOINT NAME HERE
# =========================================================
ACTIVE_SAGEMAKER_ENDPOINT = "Generated Endpoint Name"

# ---------------------------------------------------------
# UI Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube AI Summarizer | AWS SageMaker",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 YouTube Video AI Summarizer")

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cloud Configuration")
    
    endpoint_name = st.text_input(
        "SageMaker Endpoint Name",
        value=ACTIVE_SAGEMAKER_ENDPOINT,
        help="Make sure this matches the active endpoint name in your AWS SageMaker console."
    )
    aws_region = st.selectbox(
        "AWS Region",
        options=["us-east-1", "us-east-2", "us-west-2", "eu-west-1", "ap-south-1"],
        index=0
    )
    model_type = st.selectbox(
        "Model Architecture",
        options=["LLM / Text-Generation (Mistral, Llama, Falcon, Gemma)", "Summarizer Pipeline (BART, Pegasus, T5)"],
        index=0,
        help="Select LLM for decoder models, or Summarizer for Seq2Seq pipelines."
    )
    
    st.divider()
    st.subheader("🎛️ Summarization Controls")
    max_chunk_words = st.slider("Chunk Size (Words)", 250, 600, 400, step=50)
    max_tokens = st.slider("Max Summary Length per Chunk", 60, 250, 130)
    min_tokens = st.slider("Min Summary Length per Chunk", 20, 80, 30)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def extract_video_id(url: str) -> str:
    """Extracts the 11-character YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/embed\/|\/v\/|youtu\.be\/|\/shorts\/|^)([0-9A-Za-z_-]{11})(?:[?&]|$)',
        r'(?:v=)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return ""

def fetch_youtube_transcript(video_id: str) -> str:
    """Fetches transcripts with fallbacks for auto-generated captions and object/dict compatibility."""
    try:
        # Support both classmethod (< 1.0) and instance-based (>= 1.0) APIs
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except (AttributeError, TypeError):
            transcript_list = YouTubeTranscriptApi().list(video_id)

        # 1. Try manually created English
        try:
            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
        except Exception:
            # 2. Fall back to auto-generated English
            try:
                transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
            except Exception:
                # 3. Fall back to translating any available transcript to English
                first_transcript = next(iter(transcript_list))
                transcript = first_transcript.translate('en')

        transcript_data = transcript.fetch()

        # Handle both FetchedTranscriptSnippet objects (.text) and legacy dictionaries (['text'])
        text_snippets = []
        for item in transcript_data:
            if hasattr(item, "text"):
                text_snippets.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                text_snippets.append(item["text"])
            else:
                text_snippets.append(str(item))

        full_text = " ".join(text_snippets)
        
        if not full_text.strip():
            raise ValueError("YouTube returned an empty transcript.")
            
        return full_text
        
    except Exception as e:
        error_msg = str(e)
        if "no element found" in error_msg.lower() or "line 1, column 0" in error_msg.lower():
            raise ValueError(
                "YouTube rate-limited or blocked transcript retrieval from this IP. "
                "Try another video with manual English CC or verify your network."
            )
        raise ValueError(f"Could not retrieve transcript: {error_msg}")

def chunk_text(text: str, chunk_size: int = 400) -> list:
    """Splits long text into manageable word chunks for context windows."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
    return chunks

def parse_sagemaker_response(result) -> str:
    """Extracts summary text across Hugging Face pipeline schemas."""
    if isinstance(result, list) and len(result) > 0:
        first = result[0]
        if isinstance(first, dict):
            if "summary_text" in first:
                return first["summary_text"].strip()
            if "generated_text" in first:
                text = first["generated_text"].strip()
                if "Summary:" in text:
                    text = text.split("Summary:")[-1].strip()
                return text
        elif isinstance(first, str):
            return first.strip()
            
    elif isinstance(result, dict):
        if "summary_text" in result:
            return result["summary_text"].strip()
        if "generated_text" in result:
            text = result["generated_text"].strip()
            if "Summary:" in text:
                text = text.split("Summary:")[-1].strip()
            return text
            
    return str(result)

def summarize_chunk_sagemaker(client, endpoint: str, text_chunk: str, max_len: int, min_len: int, is_llm: bool) -> str:
    """Invokes SageMaker with clean model parameters accepted by model.generate()."""
    if is_llm:
        prompt = (
            f"You are a helpful assistant. Summarize the key points of this video transcript section concisely:\n\n"
            f"{text_chunk}\n\n"
            f"Summary:"
        )
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_len,
                "do_sample": False
            }
        }
    else:
        # Seq2Seq Summarizer (BART, Pegasus, T5)
        inputs = f"summarize: {text_chunk}" if "t5" in endpoint.lower() else text_chunk
        payload = {
            "inputs": inputs,
            "parameters": {
                "max_new_tokens": max_len,
                "min_length": min_len,
                "do_sample": False
            }
        }

    response = client.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    return parse_sagemaker_response(result)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
col_main, col_preview = st.columns([3, 2])

with col_main:
    youtube_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    generate_btn = st.button("🚀 Fetch & Summarize Video", type="primary")

video_id = extract_video_id(youtube_url) if youtube_url else ""

with col_preview:
    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")

if generate_btn:
    if not video_id:
        st.error("Please enter a valid YouTube video URL.")
    elif not endpoint_name.strip():
        st.error("Please provide an active SageMaker Endpoint Name in the sidebar.")
    else:
        try:
            with st.spinner("Extracting transcript from YouTube..."):
                raw_transcript = fetch_youtube_transcript(video_id)
                word_count = len(raw_transcript.split())
                chunks = chunk_text(raw_transcript, chunk_size=max_chunk_words)

            st.info(f"📊 Transcript Loaded: **{word_count} words** split into **{len(chunks)} chunk(s)** for model inference.")

            # Initialize Boto3 Runtime Client
            runtime_client = boto3.client("sagemaker-runtime", region_name=aws_region)
            summaries = []

            # Progress Bar for Chunk Inference
            progress_bar = st.progress(0)
            status_text = st.empty()
            is_llm_mode = "LLM" in model_type

            for idx, chunk in enumerate(chunks):
                status_text.text(f"Processing chunk {idx + 1} of {len(chunks)} on SageMaker...")
                summary_part = summarize_chunk_sagemaker(
                    client=runtime_client,
                    endpoint=endpoint_name,
                    text_chunk=chunk,
                    max_len=max_tokens,
                    min_len=min_tokens,
                    is_llm=is_llm_mode
                )
                summaries.append(summary_part)
                progress_bar.progress((idx + 1) / len(chunks))

            status_text.empty()
            progress_bar.empty()

            # Display Final Summaries
            st.subheader("📝 Executive Summary & Key Takeaways")
            for i, part in enumerate(summaries, start=1):
                if len(chunks) > 1:
                    st.markdown(f"**Section {i}:**")
                st.write(part)

            with st.expander("📄 View Full Raw Transcript"):
                st.write(raw_transcript)

        except ClientError as ce:
            st.error(f"AWS SageMaker Error ({ce.response['Error']['Code']}): {ce.response['Error']['Message']}")
        except NoCredentialsError:
            st.error("AWS credentials not detected. Run `aws configure` in your terminal or set AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY.")
        except Exception as err:
            st.error(f"Execution Error: {str(err)}")
