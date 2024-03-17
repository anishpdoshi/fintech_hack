"""
Demonstrates how to use the `ChatInterface` to create a chatbot using
[LangChain Expression Language](https://python.langchain.com/docs/expression_language/) (LCEL)
with streaming and memory.
"""

from operator import itemgetter

import panel as pn
from huggingface_hub import hf_hub_download
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms.llamacpp import LlamaCpp

pn.extension()

REPO_ID = "TheBloke/zephyr-7B-beta-GGUF"
FILENAME = "zephyr-7b-beta.Q5_K_M.gguf"
SYSTEM_PROMPT = "Try to be a silly comedian."

def load_llm(
        model_path="/Users/apdoshi/llama.cpp/models/llama-2-13b.Q5_K_M.gguf",
        repo_id: str = REPO_ID,
        filename: str = FILENAME,
        **kwargs
    ):
    if model_path is None:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    llm = LlamaCpp(model_path=model_path, **kwargs)
    return llm


def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    message = ""
    inputs = {"input": contents}
    for token in chain.stream(inputs):
        message += token
        yield message
    print(message)
    memory.save_context(inputs, {"output": message})


model = load_llm(
    repo_id=REPO_ID,
    filename=FILENAME,
    streaming=True,
    n_gpu_layers=1,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
)
memory = ConversationSummaryBufferMemory(return_messages=True, llm=model)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
output_parser = StrOutputParser()
chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
    | output_parser
)
print("HERE")

chat_interface = pn.chat.ChatInterface(
    pn.chat.ChatMessage(
        "Offer a topic and Mistral will try to be funny!", user="System"
    ),
    callback=callback,
    callback_user="Mistral",
)
chat_interface.servable()