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
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

huggingface_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
huggingface_llm = ChatHuggingFace(llm=huggingface_llm)

#NVIDIA
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# nvidia_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
# nvidia_llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
# nvidia_llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
nvidia_llm = ChatNVIDIA(model="google/gemma-2-27b-it")