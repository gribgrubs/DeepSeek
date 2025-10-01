from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import os
import json
from typing import AsyncGenerator

app = FastAPI()

# NVIDIA NIM API configuration
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

@app.get("/")
async def root():
    return {"status": "OpenAI to NVIDIA NIM Proxy is running"}

@app.get("/v1/models")
async def list_models():
    """List available models"""
    async with httpx.AsyncClient() as client:
        try:
            headers = {
                "Authorization": f"Bearer {NIM_API_KEY}",
                "Content-Type": "application/json"
            }
            response = await client.get(f"{NIM_BASE_URL}/models", headers=headers)
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions to NVIDIA NIM"""
    try:
        body = await request.json()
        
        # Transform OpenAI format to NIM format if needed
        nim_body = {
            "model": body.get("model", "meta/llama-3.1-8b-instruct"),
            "messages": body.get("messages", []),
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 1.0),
            "max_tokens": body.get("max_tokens", 1024),
            "stream": body.get("stream", False)
        }
        
        headers = {
            "Authorization": f"Bearer {NIM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            if nim_body["stream"]:
                return StreamingResponse(
                    stream_nim_response(client, nim_body, headers),
                    media_type="text/event-stream"
                )
            else:
                response = await client.post(
                    f"{NIM_BASE_URL}/chat/completions",
                    json=nim_body,
                    headers=headers
                )
                return JSONResponse(content=response.json())
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_nim_response(
    client: httpx.AsyncClient, 
    body: dict, 
    headers: dict
) -> AsyncGenerator[str, None]:
    """Stream responses from NVIDIA NIM"""
    async with client.stream(
        "POST",
        f"{NIM_BASE_URL}/chat/completions",
        json=body,
        headers=headers
    ) as response:
        async for chunk in response.aiter_bytes():
            if chunk:
                yield chunk.decode("utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
