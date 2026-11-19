"""Week 1: a single LLM call over HTTP.

Sends one chat-completions request to the NVIDIA API Catalog
(OpenAI-compatible endpoint), prints the status code, the raw JSON body,
and the model's reply text.

Usage:
    export NVIDIA_API_KEY=YOUR_NVIDIA_API_KEY   # PowerShell: $env:NVIDIA_API_KEY="..."
    python week1_single_call.py
"""

import json
import os

import requests

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "minimaxai/minimax-m3.1-flash"


def call_model(messages):
    # Read the key from the environment; never hard-code it in source.
    api_key = os.environ["NVIDIA_API_KEY"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        # Reasoning models spend tokens on thinking too, so leave room for both.
        "max_tokens": 512,
        "temperature": 1,
        "top_p": 0.95,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

    # Print the status first so it is visible even if the body is not JSON.
    print("status:", response.status_code)
    data = response.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))

    if response.ok:
        print("reply:", data["choices"][0]["message"]["content"])
    else:
        print("error:", data.get("error", data))

    return data


if __name__ == "__main__":
    call_model([
        {"role": "user", "content": "用一句話說明什麼是 HTTP"},
    ])
