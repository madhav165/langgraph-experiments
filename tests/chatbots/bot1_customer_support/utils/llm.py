import getpass
import os

from dotenv import load_dotenv

load_dotenv("../../../.env")

# OpenAI

from langchain_openai import ChatOpenAI

# Initialize the OpenAI GPT-3.5 language model
openai_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
)

#NVIDIA

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

from langchain_nvidia_ai_endpoints import ChatNVIDIA

nvidia_llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
# llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")