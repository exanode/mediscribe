import os, datetime, hashlib, hmac, urllib.parse, uuid
from dotenv import load_dotenv, dotenv_values

# 🔹 Force load only THIS .env file (never auto-detect)
env_path = os.path.join(os.path.dirname(__file__), ".env")
print(f"Explicitly loading .env from: {env_path}")

if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
    vals = dotenv_values(env_path)
    print("Loaded keys from .env:", list(vals.keys()))
else:
    print("No .env found at", env_path)

# Now read after loading
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION") or "us-east-1"
SERVICE = "transcribe"

print(f"AWS_ACCESS_KEY_ID: {AWS_ACCESS_KEY}")
print(f"AWS_REGION: {AWS_REGION}")


def _sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(key, date_stamp, region, service):
    k_date = _sign(("AWS4" + key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    return _sign(k_service, "aws4_request")


def create_presigned_url(
    region=AWS_REGION,
    language="en-US",
    rate=16000,
    specialty="PRIMARYCARE",
    conv_type="CONVERSATION",
):
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise RuntimeError("Missing AWS credentials")

    try:
        rate_int = int(rate)
    except (TypeError, ValueError):
        rate_int = 16000

    if rate_int not in (8000, 16000):
        rate_int = 16000

    host = f"transcribestreaming.{region}.amazonaws.com:8443"
    endpoint = f"wss://{host}/medical-stream-transcription-websocket"

    t = datetime.datetime.utcnow()
    amz_date = t.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = t.strftime("%Y%m%d")

    credential_scope = f"{date_stamp}/{region}/{SERVICE}/aws4_request"
    credential = f"{AWS_ACCESS_KEY}/{credential_scope}"

    params = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": urllib.parse.quote(credential, safe=""),
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": "300",
        "X-Amz-SignedHeaders": "host",
        "language-code": language,
        "media-encoding": "pcm",
        "sample-rate": str(rate_int),
        "specialty": specialty,
        "type": conv_type,
        "session-id": str(uuid.uuid4()),
    }

    # Canonical querystring must be alphabetically sorted
    canonical_querystring = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    canonical_headers = f"host:{host}\n"
    signed_headers = "host"
    payload_hash = hashlib.sha256(b"").hexdigest()

    canonical_request = "\n".join(
        [
            "GET",
            "/medical-stream-transcription-websocket",
            canonical_querystring,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )

    signing_key = _get_signature_key(AWS_SECRET_KEY, date_stamp, region, SERVICE)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return f"{endpoint}?{canonical_querystring}&X-Amz-Signature={signature}"
