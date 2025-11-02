import asyncio, websockets, sounddevice as sd, numpy as np, scipy.signal, struct, zlib
from aws_signer import create_presigned_url
import sys
import json
import re


sys.stdout.reconfigure(encoding='utf-8')

def create_audio_event(audio_chunk):
    headers = {
        ':message-type': {'type': 7, 'value': 'event'},
        ':event-type': {'type': 7, 'value': 'AudioEvent'},
        ':content-type': {'type': 7, 'value': 'application/octet-stream'}
    }

    headers_buf = b''
    for name, header in headers.items():
        name_bytes = name.encode('utf-8')
        headers_buf += struct.pack('!B', len(name_bytes)) + name_bytes
        headers_buf += struct.pack('!B', header['type'])
        value_bytes = header['value'].encode('utf-8')
        headers_buf += struct.pack('!H', len(value_bytes)) + value_bytes

    headers_length = len(headers_buf)
    payload_length = len(audio_chunk)
    total_length = 16 + headers_length + payload_length

    prelude = struct.pack('!I', total_length) + struct.pack('!I', headers_length)
    prelude_crc = struct.pack('!I', zlib.crc32(prelude) & 0xffffffff)

    message = prelude + prelude_crc + headers_buf + audio_chunk
    message_crc = struct.pack('!I', zlib.crc32(message) & 0xffffffff)

    return message + message_crc

def create_end_event():
    return create_audio_event(b'')

async def main():
    url = create_presigned_url()
    print("Connecting to:", url[:120])

    sr_in = int(sd.query_devices(sd.default.device[0], 'input')['default_samplerate'])
    sr_target = 16000
    duration = 5

    print(f"Recording {duration}s from mic @ {sr_in} Hz, resampling to {sr_target} Hz...")
    audio = sd.rec(int(sr_in * duration), samplerate=sr_in, channels=1, dtype='float32')
    sd.wait()

    audio = np.nan_to_num(audio[:, 0])
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    resampled = scipy.signal.resample_poly(audio, sr_target, sr_in)
    pcm_int16 = (resampled * 32767).astype('<i2')
    pcm_bytes = pcm_int16.tobytes()

    chunk_ms = 100
    chunk_size = int(sr_target * 2 * chunk_ms / 1000)

    async with websockets.connect(url, max_size=10_000_000) as ws:
        print("Connected to AWS Transcribe streaming.")

        async def sender():
            for i in range(0, len(pcm_bytes), chunk_size):
                await ws.send(create_audio_event(pcm_bytes[i:i+chunk_size]))
                await asyncio.sleep(chunk_ms / 1000.0)
            await ws.send(create_end_event())
            await asyncio.sleep(2)
            await ws.close()
            print("Finished sending audio.")



        async def receiver():
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        msg_str = msg.decode('utf-8', errors='ignore')
                        # extract JSON part only
                        match = re.search(r'(\{.*"Transcript".*\})', msg_str)
                        if match:
                            data = json.loads(match.group(1))
                            results = data["Transcript"].get("Results", [])
                            for r in results:
                                if not r.get("IsPartial", True):
                                    alts = r.get("Alternatives", [])
                                    if alts:
                                        print("Transcript:", alts[0].get("Transcript", ""))
            except websockets.ConnectionClosed as e:
                print("AWS closed connection:", e)
                
        await asyncio.gather(sender(), receiver())


if __name__ == "__main__":
    asyncio.run(main())