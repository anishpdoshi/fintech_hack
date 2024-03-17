
import os, requests
import lmql
import json
from openai import OpenAI
import functools

client = OpenAI()

@functools.lru_cache()
def query_llm(prompt, method="together", model="togethercomputer/llama-2-70b-chat"):
    if method == "openai":
        url = "https://api.openai.com/v1/chat/completions"

        # The headers including the Authorization with the API key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer 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"  # Replace YOUR_OPENAI_API_KEY with your actual OpenAI API key
        }

        # The data payload as a dictionary
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        # Making the POST request
        response = requests.post(url, headers=headers, data=json.dumps(data))

        json_resp = response.json()
        return json_resp['choices'][0]['message']['content']

    else:
        url = 'https://api.together.xyz/inference'
        headers = {
            'Authorization': 'Bearer ' + os.environ["TOGETHER_API_KEY"],
            'accept': 'application/json',
            'content-type': 'application/json'
        }

        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1024,
            "stop": ".",
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1
        }

        response = requests.post(url, json=data, headers=headers)
        json_resp = response.json()
        return json_resp['output']['choices'][0]['text']
    