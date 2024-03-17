import os, requests

url = 'https://api.together.xyz/inference'
headers = {
    'Authorization': 'Bearer ' + os.environ["TOGETHER_API_KEY"],
    'accept': 'application/json',
    'content-type': 'application/json'
}

data = {
    "model": "togethercomputer/llama-2-70b-chat",
    "prompt": "The capital of France is",
    "max_tokens": 128,
    "stop": ".",
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1
}

response = requests.post(url, json=data, headers=headers)
print(response.json())