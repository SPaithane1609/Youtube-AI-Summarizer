# 🎬 YouTube Video AI Summarizer with Amazon SageMaker

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker%20Inference-FF9900.svg)](https://aws.amazon.com/sagemaker/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Inference%20Toolkit-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end NLP application that extracts transcripts from YouTube videos and generates multi-chunk executive summaries using Hugging Face models deployed on dedicated Amazon SageMaker real-time endpoints.

## 1. Web App UI
![alt text](<AWS Sagemaker.png>) 

## 2. YouTube video summarized
![alt text](<AWS sagemaker results.png>)


---

## 🏛️ System Architecture

```text
               +-----------------------+
               |   YouTube Video URL   |
               +-----------+-----------+
                           |
                           v
              +--------------------------+
              | youtube-transcript-api   |
              | (Auto-CC & Translation)  |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |     Text Preprocessing   |
              | (Sliding-Window Chunking)|
              +------------+-------------+
                           |
                           v
            +------------------------------+
            |  AWS SageMaker Runtime API   |
            |     (boto3 invoke_endpoint)  |
            +--------------+---------------+
                           |
                           v
            +------------------------------+
            | Hugging Face PyTorch Container|
            | (BART / T5 / Mistral / Llama) |
            +--------------+---------------+
                           |
                           v
            +------------------------------+
            |    Streamlit Web Interface   |
            |  (Interactive Dashboard)     |
            +------------------------------+
```

---

## ✨ Features

- **Multi-Format Ingestion:** Supports standard watch URLs, short links (`youtu.be`), embed URLs, and YouTube Shorts.
- **Robust Transcript Extraction:** Cascades gracefully from manual transcripts to auto-generated CC and automated English translation.
- **Adaptive Chunking Engine:** Splits long transcripts into configurable word chunks to respect context windows and prevent payload truncation.
- **Dual Architecture Compatibility:** Supports both Seq2Seq models (`facebook/bart-large-cnn`, `google/flan-t5-large`) and decoder-only LLMs (`mistralai/Mistral-7B-Instruct`, `meta-llama/Llama-3-8B-Instruct`).
- **Configurable Inference Parameters:** Dynamically adjust chunk size, maximum generation tokens, and target AWS region via the web UI.

---

## 📁 Repository Structure

```text
.
├── app.py                  # Streamlit frontend & inference client
├── deploy.py               # Infrastructure deployment script for SageMaker
├── teardown.py             # Script to delete endpoints and stop billing
├── requirements.txt        # Application dependencies
├── .gitignore              # Ignores credentials, venvs, and cache files
└── README.md               # Project documentation
```

---

## 📋 Prerequisites

- **Python 3.10+**
- **AWS Account** with an active IAM role containing SageMaker permissions:
  - `sagemaker:CreateEndpoint`
  - `sagemaker:CreateEndpointConfig`
  - `sagemaker:CreateModel`
  - `sagemaker:InvokeEndpoint`
  - `sagemaker:DeleteEndpoint`
  - `sagemaker:DeleteEndpointConfig`
  - `iam:PassRole`
- **AWS CLI** configured on your local machine:
  ```bash
  aws configure
  ```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/<your-username>/youtube-sagemaker-summarizer.git
cd youtube-sagemaker-summarizer

# Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Deploy Model to Amazon SageMaker

Export your SageMaker Execution Role ARN:

```bash
export SAGEMAKER_EXECUTION_ROLE="arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME>"
```

Run the deployment script:

```bash
python deploy.py
```

*Wait for deployment to complete (~4–7 minutes). Copy the output endpoint name (e.g., `huggingface-pytorch-inference-2024-07-03-14-42-59-347`).*

### 3. Run the Streamlit Application

```bash
export SAGEMAKER_ENDPOINT_NAME="<YOUR_ACTIVE_ENDPOINT_NAME>"
streamlit run app.py
```

1. Navigate to `http://localhost:8501` in your browser.
2. Confirm the **SageMaker Endpoint Name** and **AWS Region** in the sidebar.
3. Paste any YouTube URL and click **🚀 Fetch & Summarize Video**.

---

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| **Endpoint Name** | Dynamic / Env | Target SageMaker endpoint name. |
| **AWS Region** | `us-east-1` | Region where the endpoint is hosted. |
| **Model Architecture** | `LLM / Text-Generation` | Switches request payload between generative LLMs and Seq2Seq pipelines. |
| **Chunk Size (Words)** | `400` | Word count per slice to avoid context overflow. |
| **Max Summary Length** | `130` | Max new tokens generated per chunk. |
| **Min Summary Length** | `30` | Minimum token constraint for Seq2Seq summarizers. |

---

## 🧹 Cost Optimization & Teardown

Amazon SageMaker real-time endpoints bill by the hour while running. To delete the endpoint and stop incurring charges:

```bash
python teardown.py
```

Enter your active endpoint name when prompted.

---

## 🛠️ Troubleshooting

| Error | Root Cause | Fix |
| :--- | :--- | :--- |
| `NoCredentialsError` | Boto3 cannot find AWS credentials. | Run `aws configure` or export `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. |
| `ModelError (400): model_kwargs are not used` | Passing pipeline keys (`return_full_text`, `truncation`) directly to `model.generate()`. | Use only valid `generate()` kwargs like `max_new_tokens` and `do_sample`. |
| `FetchedTranscriptSnippet is not subscriptable` | `youtube-transcript-api` $\ge 1.0$ returns objects, not dictionaries. | Access transcript text using `item.text` instead of `item["text"]`. |
| `Could not retrieve transcript: YouTube rate-limited` | YouTube IP block on public cloud/VPN subnets. | Run the Streamlit interface locally or use a video with manual captions. |

---

## 📄 License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
