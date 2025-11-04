import sys
sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json
import struct
import zlib
import numpy as np
import websockets
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from aws_signer import create_presigned_url, AWS_REGION

print("----== app.py loaded successfully ==----")

# ------------------------------------------------------------
# Allowed configuration sets
# ------------------------------------------------------------
ALLOWED_LANGUAGES = {"en-us": "en-US"}
ALLOWED_SPECIALTIES = {
    "cardiology": "CARDIOLOGY",
    "neurology": "NEUROLOGY",
    "oncology": "ONCOLOGY",
    "primarycare": "PRIMARYCARE",
    "radiology": "RADIOLOGY",
    "urology": "UROLOGY",
    "obgyn": "OBGYN",
}
ALLOWED_CONV_TYPES = {
    "conversation": "CONVERSATION",
    "dictation": "DICTATION",
}
ALLOWED_SAMPLE_RATES = {"8000": 8000, "16000": 16000}


def _resolve_query_param(query, keys, *, default, normalizer=None, allowed=None):
    """Resolve query param with normalization and allowed-value mapping."""
    if isinstance(keys, str):
        keys = [keys]

    raw_value = None
    for key in keys:
        value = query.get(key)
        if value is not None:
            raw_value = value
            break

    if raw_value is None:
        return default

    raw_value = raw_value.strip()
    if not raw_value:
        return default

    value = normalizer(raw_value) if normalizer else raw_value

    if allowed is None:
        return value

    if isinstance(allowed, dict):
        lookup_key = value.lower()
        return allowed.get(lookup_key, default)

    if value in allowed:
        return value

    return default


MAX_COMPREHEND_TEXT_LENGTH = 20000


class AnalyzePayload(BaseModel):
    text: str


app = FastAPI(title="MediScribe Realtime", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

comprehend_client = boto3.client("comprehendmedical", region_name=AWS_REGION)


def _normalize_concept_key(value: str) -> str:
    return " ".join(value.lower().split())


def _collect_traits(traits):
    collected = []
    for trait in traits or []:
        name = trait.get("Name")
        score = trait.get("Score", 0)
        if name and score >= 0.5:
            collected.append({"name": name, "score": score})
    return collected


def _build_concept_lookup(entities, concept_key):
    lookup = {}
    for entity in entities or []:
        mention = (entity.get("Text") or "").strip()
        if not mention:
            continue
        norm = _normalize_concept_key(mention)
        concepts = []
        for concept in entity.get(concept_key, []) or []:
            code = concept.get("Code")
            if not code:
                continue
            concepts.append(
                {
                    "code": code,
                    "description": concept.get("Description"),
                    "score": concept.get("Score"),
                    "source": mention,
                }
            )
        if concepts:
            seen_codes = {c["code"] for c in lookup.get(norm, [])}
            filtered = [c for c in concepts if c["code"] not in seen_codes]
            if filtered:
                lookup.setdefault(norm, []).extend(filtered)
    return lookup


def _summarize_unique_codes(lookup):
    seen = set()
    summary = []
    for concepts in lookup.values():
        for concept in concepts:
            code = concept.get("code")
            if code and code not in seen:
                seen.add(code)
                summary.append(concept)
    return summary


@app.post("/analyze")
async def analyze(payload: AnalyzePayload):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Transcription text is required for analysis.")

    truncated = False
    if len(text) > MAX_COMPREHEND_TEXT_LENGTH:
        text = text[:MAX_COMPREHEND_TEXT_LENGTH]
        truncated = True

    try:
        detect_response = await asyncio.to_thread(comprehend_client.detect_entities_v2, Text=text)
        icd_response = await asyncio.to_thread(comprehend_client.infer_icd10_cm, Text=text)
        snomed_response = await asyncio.to_thread(comprehend_client.infer_snomedct, Text=text)
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=502, detail=f"AWS Comprehend Medical error: {exc}") from exc

    icd_lookup = _build_concept_lookup(icd_response.get("Entities"), "ICD10CMConcepts")
    snomed_lookup = _build_concept_lookup(snomed_response.get("Entities"), "SNOMEDCTConcepts")

    entities_payload = {"conditions": [], "tests": [], "treatments": [], "procedures": []}

    for entity in detect_response.get("Entities", []):
        text_value = (entity.get("Text") or "").strip()
        if not text_value:
            continue

        record = {
            "text": text_value,
            "confidence": entity.get("Score"),
            "traits": _collect_traits(entity.get("Traits")),
        }

        norm_key = _normalize_concept_key(text_value)
        if norm_key in icd_lookup:
            record["icd10"] = icd_lookup[norm_key]
        if norm_key in snomed_lookup:
            record["snomed"] = snomed_lookup[norm_key]

        category = entity.get("Category") or ""
        entity_type = (entity.get("Type") or "").upper()

        if category == "MEDICAL_CONDITION":
            entities_payload["conditions"].append(record)
        elif category == "TEST_TREATMENT_PROCEDURE":
            if "TEST" in entity_type:
                entities_payload["tests"].append(record)
            elif "TREATMENT" in entity_type:
                entities_payload["treatments"].append(record)
            else:
                entities_payload["procedures"].append(record)

    return {
        "entities": entities_payload,
        "summary": {
            "icd10": _summarize_unique_codes(icd_lookup),
            "snomed": _summarize_unique_codes(snomed_lookup),
        },
        "truncated": truncated,
    }

# ------------------------------------------------------------
# AWS EventStream Wrappers
# ------------------------------------------------------------
def create_audio_event(audio_chunk: bytes) -> bytes:
    """Wrap raw PCM audio chunk into AWS EventStream AudioEvent."""
    headers = {
        ":message-type": ("event", 7),
        ":event-type": ("AudioEvent", 7),
        ":content-type": ("application/octet-stream", 7),
    }

    headers_buf = b""
    for name, (value, htype) in headers.items():
        name_bytes = name.encode("utf-8")
        headers_buf += struct.pack("!B", len(name_bytes)) + name_bytes
        headers_buf += struct.pack("!B", htype)
        val_bytes = value.encode("utf-8")
        headers_buf += struct.pack("!H", len(val_bytes)) + val_bytes

    headers_len = len(headers_buf)
    total_len = 16 + headers_len + len(audio_chunk)

    prelude = struct.pack("!I", total_len) + struct.pack("!I", headers_len)
    prelude_crc = struct.pack("!I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + prelude_crc + headers_buf + audio_chunk
    message_crc = struct.pack("!I", zlib.crc32(message) & 0xFFFFFFFF)

    return message + message_crc


def create_end_event() -> bytes:
    """End-of-stream empty event."""
    return create_audio_event(b"")


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/")
async def home() -> HTMLResponse:
    import os
    possible_paths = [
        "templates/index.html",
        "index.html",
        os.path.join(os.path.dirname(__file__), "templates", "index.html"),
        os.path.join(os.path.dirname(__file__), "index.html"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found index.html at: {path}")
            with open(path, encoding="utf-8") as f:
                return HTMLResponse(f.read())
    
    return HTMLResponse("<h1>Error: index.html not found</h1>", status_code=500)


# ------------------------------------------------------------
# WebSocket Handler
# ------------------------------------------------------------
@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    print("=" * 60)
    print("WebSocket accepted from browser")

    # --------------------------------------------------------
    # Resolve stream configuration from browser query params
    # --------------------------------------------------------
    language = _resolve_query_param(
        ws.query_params,
        "language",
        default="en-US",
        normalizer=lambda v: v.replace("_", "-") if v else v,
        allowed=ALLOWED_LANGUAGES,
    )
    specialty = _resolve_query_param(
        ws.query_params,
        "specialty",
        default="PRIMARYCARE",
        normalizer=str.lower,
        allowed=ALLOWED_SPECIALTIES,
    )
    conv_type = _resolve_query_param(
        ws.query_params,
        ["conversationType", "convType"],
        default="CONVERSATION",
        normalizer=str.lower,
        allowed=ALLOWED_CONV_TYPES,
    )
    sample_rate = _resolve_query_param(
        ws.query_params,
        ["sampleRate", "sample-rate"],
        default="16000",
        normalizer=lambda v: "".join(ch for ch in v if ch.isdigit()),
        allowed=ALLOWED_SAMPLE_RATES,
    )
    aws_sample_rate = (
        ALLOWED_SAMPLE_RATES.get(str(sample_rate), 16000)
        if isinstance(sample_rate, str)
        else sample_rate
    )

    print(
        "Using stream configuration:",
        {
            "language": language,
            "specialty": specialty,
            "type": conv_type,
            "sample_rate": aws_sample_rate,
        },
    )

    # --------------------------------------------------------
    # Connect to AWS Transcribe Medical WebSocket
    # --------------------------------------------------------
    async def connect_to_aws():
        """Create a fresh signed AWS websocket connection."""
        signed_url = create_presigned_url(
            language=language,
            specialty=specialty,
            conv_type=conv_type,
            rate=aws_sample_rate,
        )
        print("Connecting to AWS Transcribe Medical...")
        print(f"Signed URL (first 150 chars): {signed_url[:150]}...")
        print(f"Full URL length: {len(signed_url)} characters")

        print("Attempting AWS WebSocket connection...")
        aws_ws = await asyncio.wait_for(
            websockets.connect(
                signed_url,
                ping_interval=15,   # more forgiving interval
                ping_timeout=30,    # relaxed timeout
                close_timeout=10,
                max_size=10_000_000,
            ),
            timeout=20.0,
        )
        print("✅ Connected to AWS Transcribe Medical WebSocket!")
        return aws_ws

    aws_ws = await connect_to_aws()

    # --------------------------------------------------------
    # Client → AWS: send audio
    # --------------------------------------------------------
    async def client_to_aws() -> None:
        """Receive PCM/float32 audio from browser and forward to AWS."""
        chunk_count = 0
        try:
            print("📡 Starting client_to_aws task...")
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=30.0)

                    # Handle browser disconnects or text heartbeats
                    if isinstance(msg, dict) and msg.get("type") == "websocket.disconnect":
                        print("⚠️ Browser disconnected.")
                        break
                    elif isinstance(msg, str) or (isinstance(msg, dict) and msg.get("text")):
                        text_data = msg if isinstance(msg, str) else msg.get("text", "")
                        if text_data == "ping":
                            print("💓 Received heartbeat ping from browser")
                            continue
                        else:
                            print(f"📜 Received unexpected text: {text_data}")
                            continue
                    elif isinstance(msg, dict) and "bytes" in msg:
                        data = msg["bytes"]
                    else:
                        data = msg if isinstance(msg, (bytes, bytearray)) else None

                    if not data:
                        continue

                    chunk_count += 1
                    if chunk_count == 1:
                        print(f"🎧 First chunk received: {len(data)} bytes")
                    elif chunk_count % 10 == 0:
                        print(f"🎧 Chunk {chunk_count}: {len(data)} bytes")

                    # --- Proper decoding for Float32 or Int16 ---
                    if len(data) == 0:
                        continue

                    if len(data) % 4 == 0:
                        floats = np.frombuffer(data, dtype="<f4")
                        if floats.size == 0:
                            continue
                        floats = np.nan_to_num(floats, nan=0.0, posinf=0.0, neginf=0.0)
                        rms = np.sqrt(np.mean(floats ** 2))
                        peak = np.max(np.abs(floats))
                        pcm16 = (np.clip(floats, -1, 1) * 32767).astype("<i2").tobytes()
                    elif len(data) % 2 == 0:
                        i16 = np.frombuffer(data, dtype="<i2")
                        if i16.size == 0:
                            continue
                        floats = (i16.astype(np.float32) / 32767.0)
                        rms = np.sqrt(np.mean(floats ** 2))
                        peak = np.max(np.abs(floats))
                        pcm16 = data
                    else:
                        pcm16 = data
                        rms = 0.0
                        peak = 0.0
                    # -------------------------------------------

                    event = create_audio_event(pcm16)
                    await aws_ws.send(event)

                    if chunk_count == 1:
                        print(f"✅ First audio event sent to AWS: {len(event)} bytes (PCM: {len(pcm16)} bytes)")
                        print(f"   Audio levels: RMS={rms:.4f}, Peak={peak:.4f}")
                    elif chunk_count % 20 == 0:
                        print(f"   Audio levels at chunk {chunk_count}: RMS={rms:.4f}, Peak={peak:.4f}")

                except asyncio.TimeoutError:
                    print("⏱️ No audio received for 30s, continuing...")
                    continue
                except Exception as e:
                    print(f"❌ Error in client_to_aws loop: {type(e).__name__}: {e}")
                    break

        except Exception as e:
            print(f"❌ client_to_aws error: {type(e).__name__}: {e}")
        finally:
            print(f"📊 Total audio chunks sent to AWS: {chunk_count}")
            try:
                await aws_ws.send(create_end_event())
                print("✅ Sent end-of-stream event to AWS")
            except Exception as e:
                print(f"⚠️ Failed to send end event: {e}")
            print(f"client_to_aws finished normally (ws state: {ws.client_state})")

    # --------------------------------------------------------
    # AWS → Client: forward transcripts
    # --------------------------------------------------------
    async def aws_to_client() -> None:
        """Forward messages from AWS to the browser."""
        print("📡 Starting aws_to_client task...")
        response_count = 0
        try:
            async for message in aws_ws:
                response_count += 1
                print(f"📩 Received message #{response_count} from AWS: {len(message)} bytes")

                if isinstance(message, (bytes, bytearray)):
                    try:
                        headers, payload = parse_event(message)
                        text = payload.decode("utf-8", errors="ignore")
                        if response_count <= 3 or "Transcript" in text:
                            print(f"AWS response: {text[:500]}")
                        await ws.send_text(text)
                    except Exception as e:
                        print(f"⚠️ Parse error: {type(e).__name__}: {e}")
                else:
                    print(f"Non-binary message: {message}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"🔌 AWS connection closed: code={e.code}, reason={e.reason}")
        except Exception as exc:
            print(f"❌ aws_to_client error: {type(exc).__name__}: {exc}")
        finally:
            print(f"📊 Total AWS responses received: {response_count}")
            print(f"aws_to_client finished normally (aws_ws open: {not aws_ws.closed})")
            try:
                await ws.send_text(json.dumps({"status": "aws_disconnected"}))
            except:
                pass

    # --------------------------------------------------------
    # Run bidirectional tasks with automatic AWS reconnect
    # --------------------------------------------------------
    browser_alive = True
    while browser_alive:
        try:
            print("🚀 Starting bidirectional stream tasks...")
            await asyncio.gather(client_to_aws(), aws_to_client())
            print("⚙️ Stream tasks ended; checking connection states...")

            if ws.client_state.name != "CONNECTED":
                browser_alive = False
                print("🧹 Browser disconnected, ending session.")
                break

            print("⚠️ AWS connection closed — reconnecting in 2 s...")
            await asyncio.sleep(2)
            aws_ws = await connect_to_aws()
            continue

        except websockets.exceptions.ConnectionClosed:
            print("⚠️ AWS connection closed (exception) — reconnecting in 2 s...")
            await asyncio.sleep(2)
            try:
                aws_ws = await connect_to_aws()
                continue
            except Exception as e:
                print(f"❌ Reconnection failed: {e}")
                await ws.send_text(json.dumps({"error": "AWS reconnect failed"}))
                break

    # --------------------------------------------------------
    # Clean shutdown
    # --------------------------------------------------------
    try:
        await aws_ws.close()
        print("🔌 AWS WebSocket closed")
    except Exception:
        pass

    try:
        await ws.close()
        print("🔌 Browser WebSocket closed")
    except Exception:
        pass

    print("=" * 60)


# ------------------------------------------------------------
# AWS Event Parsing
# ------------------------------------------------------------
def parse_event(message: bytes) -> tuple[bytes, bytes]:
    """Safely parse AWS event-stream messages."""
    if len(message) < 12:
        return b"", b""

    try:
        total_length, headers_length = struct.unpack("!II", message[:8])
    except Exception:
        return b"", b""

    if total_length > len(message) - 4:
        total_length = len(message) - 4

    headers = message[8 : 8 + headers_length]
    payload = message[8 + headers_length : -4] if headers_length < len(message) else b""

    return headers, payload
