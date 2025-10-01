from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import os
import json

app = FastAPI()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

@app.get("/")
async def root():
    return {"status": "DeepSeek Proxy is running!", "models": ["deepseek-chat", "deepseek-reasoner"]}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        
        deepseek_body = {
            "model": body.get("model", "deepseek-chat"),
            "messages": body.get("messages", []),
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 1.0),
            "max_tokens": body.get("max_tokens", 2048),
            "stream": body.get("stream", False)
        }
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            if deepseek_body["stream"]:
                return StreamingResponse(
                    stream_response(client, deepseek_body, headers),
                    media_type="text/event-stream"
                )
            else:
                response = await client.post(
                    f"{DEEPSEEK_BASE_URL}/chat/completions",
                    json=deepseek_body,
                    headers=headers
                )
                return JSONResponse(content=response.json())
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(client, body, headers):
    async with client.stream(
        "POST",
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        json=body,
        headers=headers
    ) as response:
        async for chunk in response.aiter_bytes():
            if chunk:
                yield chunk

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
