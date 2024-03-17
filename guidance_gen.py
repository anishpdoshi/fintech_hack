# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="dhruvabansal/llama-2-13b")

pipe("Hello, ")