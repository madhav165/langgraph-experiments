import getpass
import os

from dotenv import load_dotenv

load_dotenv("../../.env")

# OpenAI
from langchain_openai import ChatOpenAI

openai_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
)

#HuggingFace
from langchain_huggingface import ChatHuggingFace


#NVIDIA
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# nvidia_llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
nvidia_llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
# nvidia_llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
# nvidia_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")