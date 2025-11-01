import asyncio
import json
import struct
import zlib

import websockets
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from aws_signer import create_presigned_url

app = FastAPI(title="MediScribe Realtime", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home() -> HTMLResponse:
    with open("templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    signed_url = create_presigned_url()

    try:
        async with websockets.connect(
            signed_url, ping_interval=10, ping_timeout=20, close_timeout=10
        ) as aws_ws:

            async def client_to_aws() -> None:
                try:
                    while True:
                        data = await ws.receive_bytes()
                        await aws_ws.send(data)
                except Exception:
                    await aws_ws.send(b"")
                    await aws_ws.close()

            async def aws_to_client() -> None:
                try:
                    async for message in aws_ws:
                        if isinstance(message, (bytes, bytearray)):
                            _, payload = parse_event(message)
                            await ws.send_text(payload.decode("utf-8"))
                except Exception as exc:
                    await ws.send_text(json.dumps({"error": str(exc)}))

            await asyncio.gather(client_to_aws(), aws_to_client())
    except Exception as exc:
        await ws.send_text(json.dumps({"error": str(exc)}))
    finally:
        await ws.close()


def parse_event(message: bytes) -> tuple[bytes, bytes]:
    total_length, headers_length = struct.unpack("!II", message[:8])
    if total_length != len(message) - 4:
        raise ValueError("Unexpected AWS event stream length")

    headers = message[8 : 8 + headers_length]
    payload = message[8 + headers_length : -4]
    expected_crc, = struct.unpack("!I", message[-4:])
    actual_crc = zlib.crc32(message[:-4]) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise ValueError("CRC mismatch from AWS event stream message")
    return headers, payload
