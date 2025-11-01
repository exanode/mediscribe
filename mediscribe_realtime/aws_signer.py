import datetime
import hashlib
import hmac
import os
import urllib.parse

from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SERVICE = "transcribe"


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(key: str, date: str, region: str, service: str) -> bytes:
    k_date = _sign(("AWS4" + key).encode("utf-8"), date)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    return _sign(k_service, "aws4_request")


def create_presigned_url(
    *,
    region: str = AWS_REGION,
    language: str = "en-US",
    specialty: str = "PRIMARYCARE",
    rate: int = 16000,
) -> str:
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise RuntimeError("Missing AWS credentials. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

    host = f"transcribestreaming.{region}.amazonaws.com:8443"
    endpoint = f"wss://{host}/medical-stream-transcription-websocket"

    timestamp = datetime.datetime.utcnow()
    amz_date = timestamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = timestamp.strftime("%Y%m%d")

    params = {
        "language-code": language,
        "media-encoding": "pcm",
        "sample-rate": str(rate),
        "specialty": specialty,
        "type": "CONVERSATION",
    }

    canonical_querystring = "&".join(
        f"{urllib.parse.quote(key)}={urllib.parse.quote(value)}" for key, value in sorted(params.items())
    )

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

    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/{SERVICE}/aws4_request"

    string_to_sign = "\n".join(
        [
            algorithm,
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )

    signing_key = _get_signature_key(AWS_SECRET_KEY, date_stamp, region, SERVICE)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    canonical_querystring += (
        f"&X-Amz-Algorithm={algorithm}"
        f"&X-Amz-Credential={urllib.parse.quote_plus(AWS_ACCESS_KEY + '/' + credential_scope)}"
        f"&X-Amz-Date={amz_date}"
        f"&X-Amz-Expires=300"
        f"&X-Amz-SignedHeaders={signed_headers}"
        f"&X-Amz-Signature={signature}"
    )

    return f"{endpoint}?{canonical_querystring}"
