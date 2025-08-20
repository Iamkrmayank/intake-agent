# app.py
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st
import requests

# ---- Optional Azure Document Intelligence (OCR) ----
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
except Exception:
    AzureKeyCredential = None
    DocumentIntelligenceClient = None


# =========================
# SECRETS / CONFIG HELPERS
# =========================
def sget(path: str, default: Optional[str] = None) -> Optional[str]:
    """
    Nested fetch from st.secrets with dot-path, e.g. sget("azure.AZURE_API_KEY").
    """
    try:
        node = st.secrets
        for part in path.split("."):
            node = node[part]
        return node
    except Exception:
        return default


# ---- Read all config from st.secrets (no env vars!) ----
AZURE_DI_ENDPOINT = sget("azure.AZURE_DI_ENDPOINT", "")
AZURE_API_KEY = sget("azure.AZURE_API_KEY", "")

AZURE_OPENAI_ENDPOINT = sget("azure_openai.AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = sget("azure_openai.AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = sget("azure_openai.AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
GPT_DEPLOYMENT = sget("azure_openai.GPT_DEPLOYMENT", "gpt-4")  # Azure *deployment* name


# =========================
# OPTIONAL CLIENT BOOT
# =========================
di_client = None
if DocumentIntelligenceClient and AzureKeyCredential and AZURE_DI_ENDPOINT and AZURE_API_KEY:
    try:
        di_client = DocumentIntelligenceClient(
            endpoint=AZURE_DI_ENDPOINT,
            credential=AzureKeyCredential(AZURE_API_KEY),
        )
    except Exception as e:
        di_client = None
        st.sidebar.warning(f"Document Intelligence client not initialized: {e}")


# =========================
# UTILITIES (no-PHI logging)
# =========================
def generate_claim_id() -> str:
    return str(uuid.uuid4())

def iso_date_only(dt: str) -> str:
    if not dt:
        return dt
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", dt)
    return "-".join(m.groups()) if m else dt

def norm_gender(g: Optional[str]) -> str:
    if not g:
        return "U"
    g = g.strip().lower()
    if g in ["m", "male"]:
        return "M"
    if g in ["f", "female"]:
        return "F"
    if g in ["o", "other", "non-binary", "nonbinary", "nb"]:
        return "O"
    return "U"

def validate_npi(value: Optional[str]) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    return digits if len(digits) == 10 else ""

def coerce_schema(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "claim_id": raw.get("claim_id", generate_claim_id()),
        "patient": {
            "mrn": str(raw.get("patient", {}).get("mrn", "")) or "",
            "first_name": raw.get("patient", {}).get("first_name", "") or "",
            "last_name": raw.get("patient", {}).get("last_name", "") or "",
            "dob": iso_date_only(raw.get("patient", {}).get("dob", "")) or "",
            "sex": norm_gender(raw.get("patient", {}).get("sex", "")),
            "payer_id": raw.get("patient", {}).get("payer_id", "") or "",
            "payer_name": raw.get("patient", {}).get("payer_name", "") or "",
            "member_id": raw.get("patient", {}).get("member_id", "") or "",
            "group_id": raw.get("patient", {}).get("group_id", "") or ""
        },
        "encounter": {
            "date": iso_date_only(raw.get("encounter", {}).get("date", "")) or "",
            "site": (raw.get("encounter", {}).get("site", "") or "").strip() or "",
            "provider_npi": validate_npi(raw.get("encounter", {}).get("provider_npi", "")),
            "location_npi": validate_npi(raw.get("encounter", {}).get("location_npi", "")),
        },
        "documents": [],
        "clinical_note": {
            "doc_id": "",
            "blob_url": "",
            "text_preview": ""
        },
        "status": raw.get("status", "PARSING"),
        "meta": {
            "source": raw.get("meta", {}).get("source", "upload"),
            "created_at": raw.get("meta", {}).get("created_at", datetime.utcnow().isoformat() + "Z"),
            "confidence": float(raw.get("meta", {}).get("confidence", 0.85))
        }
    }

    # documents normalization
    docs_in = raw.get("documents", []) or []
    docs_norm = []
    for d in docs_in:
        doc_type = d.get("type", d.get("doc_type", "")).strip().lower()
        docs_norm.append({
            "doc_id": d.get("doc_id", str(uuid.uuid4())),
            "type": doc_type if doc_type else "",
            "blob_url": d.get("blob_url", d.get("signed_url", "")) or ""
        })
    out["documents"] = docs_norm

    # clinical_note normalization
    cn = raw.get("clinical_note", {})
    if not cn and docs_norm:
        for d in docs_norm:
            if d.get("type") in ["clinical_note", "note"]:
                cn = {
                    "doc_id": d.get("doc_id", ""),
                    "blob_url": d.get("blob_url", ""),
                    "text_preview": ""
                }
                break
    out["clinical_note"] = {
        "doc_id": cn.get("doc_id", ""),
        "blob_url": cn.get("blob_url", ""),
        "text_preview": (cn.get("text_preview", "") or "")[:240]
    }

    # Human-in-loop rule
    missing_crit = any([
        out["patient"]["dob"] == "",
        out["patient"]["member_id"] == "",
        out["encounter"]["provider_npi"] == ""
    ])
    if out["meta"]["confidence"] < 0.7 or missing_crit:
        out["status"] = "PARSING"

    return out


def build_extraction_prompt(input_payload: Dict[str, Any]) -> str:
    return f"""
You are the Gooclaim Intake Agent. Be compliance-first (HIPAA/SOC2/ABDM/IRDAI). Do NOT include PHI in commentary.
Task: Read the provided intake JSON (which may contain signed URLs and optional hints/FHIR-like fields), and output ONLY a JSON
object strictly matching the ClaimEnvelope schema below. If a field is unknown, output an empty string "" (not null).

ClaimEnvelope schema:
{{
  "claim_id": "uuid (YOU MAY leave empty; server will assign)",
  "patient": {{
    "mrn": "string",
    "first_name": "string",
    "last_name": "string",
    "dob": "YYYY-MM-DD",
    "sex": "M|F|O|U",
    "payer_id": "string",
    "payer_name": "string",
    "member_id": "string",
    "group_id": "string"
  }},
  "encounter": {{
    "date": "YYYY-MM-DD",
    "site": "OP|IP|ER|Tele",
    "provider_npi": "string(10 digits) or empty",
    "location_npi": "string(10 digits) or empty"
  }},
  "documents": [
    {{"doc_id":"uuid-or-stable-id","type":"intake_form|insurance_card|clinical_note|note","blob_url":"signed-url-or-ref"}}
  ],
  "clinical_note": {{
    "doc_id": "uuid-or-stable-id or empty",
    "blob_url": "signed-url-or-ref or empty",
    "text_preview": "short non-sensitive snippet or empty"
  }},
  "status": "PARSING|READY|ERROR",
  "meta": {{
    "source": "upload|api|email|fhir",
    "created_at": "ISO8601",
    "confidence": 0.0
  }}
}}

Input:
{json.dumps(input_payload, indent=2)}
ONLY return the JSON. No extra commentary.
    """.strip()


# ---- Azure OpenAI via REST (from secrets) ----
def azure_chat(messages, *, temperature=0.0, max_tokens=1800, force_json=True) -> str:
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and GPT_DEPLOYMENT):
        raise RuntimeError(
            "Azure OpenAI config missing. Please set azure_openai.AZURE_OPENAI_ENDPOINT, "
            "azure_openai.AZURE_OPENAI_API_KEY, and azure_openai.GPT_DEPLOYMENT in st.secrets."
        )

    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}
    url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{GPT_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_OPENAI_API_VERSION}
    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}

    r = requests.post(url, headers=headers, params=params, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_gpt_for_envelope(input_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = build_extraction_prompt(input_payload)
    content = azure_chat(
        messages=[
            {"role": "system", "content": "You are a compliance-focused Gooclaim Intake Agent. Return ONLY JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=2000,
        force_json=True
    )

    # Parse JSON strictly; if model adds text, try to extract
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{(?:[^{}]|(?R))*\}", content, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def build_demo_sample() -> Dict[str, Any]:
    return {
        "clinic_id": "CLINIC_42",
        "source": "upload",
        "uploader_user_id": "U_983",
        "documents": [
            {
                "filename": "intake_form.pdf",
                "doc_type": "intake_form",
                "mime": "application/pdf",
                "signed_url": "https://secure.gooclaim.com/docs/123456?sig=abc123xyz"
            },
            {
                "filename": "insurance_card.jpg",
                "doc_type": "insurance_card",
                "mime": "image/jpeg",
                "signed_url": "https://secure.gooclaim.com/docs/789101?sig=def456uvw"
            },
            {
                "filename": "clinical_note.pdf",
                "doc_type": "clinical_note",
                "mime": "application/pdf",
                "signed_url": "https://secure.gooclaim.com/docs/112233?sig=ghi789rst"
            }
        ],
        "hints": {
            "patient": {
                "mrn": "MRN123456",
                "first_name": "John",
                "last_name": "Doe",
                "dob": "1985-06-15",
                "sex": "M"
            },
            "encounter": {
                "date": "2025-08-10",
                "site": "OP",
                "provider_npi": "1234567890",
                "location_npi": "0987654321"
            },
            "payer": {
                "payer_id": "BCBS001",
                "payer_name": "Blue Cross Blue Shield",
                "member_id": "M123456789",
                "group_id": "G987654"
            },
            "clinical_note_preview": "Patient reports fever and cough for 3 days; mild wheeze on exam. Plan: chest X-ray, azithromycin."
        }
    }


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Gooclaim Intake Agent", page_icon="🏥", layout="wide")
st.title("🏥 Gooclaim Intake Agent")
st.caption("Upload or paste JSON → Normalize to ClaimEnvelope (compliance-first demo)")

with st.sidebar:
    st.header("⚙️ Input Mode")
    mode = st.radio("Choose Input Type:", ["Upload JSON", "Paste JSON", "Load Demo Sample"])
    st.markdown("**Note:** No PHI is logged. Only local processing for this demo.")

    st.divider()
    st.subheader("🔐 Secrets Status")
    def ok(x: bool) -> str:
        return "✅" if x else "⚠️"

    st.write(f"{ok(bool(AZURE_OPENAI_ENDPOINT))} `azure_openai.AZURE_OPENAI_ENDPOINT`")
    st.write(f"{ok(bool(AZURE_OPENAI_API_KEY))} `azure_openai.AZURE_OPENAI_API_KEY`")
    st.write(f"{ok(bool(GPT_DEPLOYMENT))} `azure_openai.GPT_DEPLOYMENT`")
    st.write(f"{ok(bool(AZURE_DI_ENDPOINT and AZURE_API_KEY))} Azure Document Intelligence (optional)")

json_payload: Optional[Dict[str, Any]] = None

if mode == "Upload JSON":
    up = st.file_uploader("Upload sample_input.json", type=["json"])
    if up:
        try:
            json_payload = json.load(up)
            st.subheader("Input JSON")
            st.json(json_payload)
        except Exception as e:
            st.error(f"Invalid JSON file: {e}")

elif mode == "Paste JSON":
    pasted = st.text_area("Paste JSON here", height=260, placeholder="Paste your JSON (signed URLs or FHIR-ish)...")
    if pasted.strip():
        try:
            json_payload = json.loads(pasted)
            st.subheader("Input JSON")
            st.json(json_payload)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

else:  # Load Demo Sample
    json_payload = build_demo_sample()
    st.subheader("Demo Input JSON")
    st.json(json_payload)

st.markdown("---")

if json_payload and st.button("🚀 Process Intake"):
    with st.spinner("Extracting & normalizing ClaimEnvelope..."):
        # 1) GPT extraction
        gpt_out = None
        try:
            gpt_out = call_gpt_for_envelope(json_payload)
        except Exception as e:
            st.warning(f"Model extraction failed, will try fallback from hints. Details: {e}")

        # 2) Fallback from hints if GPT couldn’t be parsed
        if not gpt_out:
            hints = json_payload.get("hints", {})
            gpt_out = {
                "patient": {
                    "mrn": hints.get("patient", {}).get("mrn", ""),
                    "first_name": hints.get("patient", {}).get("first_name", ""),
                    "last_name": hints.get("patient", {}).get("last_name", ""),
                    "dob": hints.get("patient", {}).get("dob", ""),
                    "sex": hints.get("patient", {}).get("sex", ""),
                    "payer_id": hints.get("payer", {}).get("payer_id", ""),
                    "payer_name": hints.get("payer", {}).get("payer_name", ""),
                    "member_id": hints.get("payer", {}).get("member_id", ""),
                    "group_id": hints.get("payer", {}).get("group_id", "")
                },
                "encounter": {
                    "date": hints.get("encounter", {}).get("date", ""),
                    "site": hints.get("encounter", {}).get("site", ""),
                    "provider_npi": hints.get("encounter", {}).get("provider_npi", ""),
                    "location_npi": hints.get("encounter", {}).get("location_npi", "")
                },
                "documents": [
                    {
                        "doc_id": f"d{i+1}",
                        "type": d.get("doc_type", ""),
                        "blob_url": d.get("signed_url", "")
                    } for i, d in enumerate(json_payload.get("documents", []))
                ],
                "clinical_note": {
                    "doc_id": "d3" if len(json_payload.get("documents", [])) >= 3 else "",
                    "blob_url": json_payload.get("documents", [{}])[2].get("signed_url", "") if len(json_payload.get("documents", [])) >= 3 else "",
                    "text_preview": hints.get("clinical_note_preview", "")
                },
                "status": "PARSING",
                "meta": {"source": json_payload.get("source", "upload"), "created_at": datetime.utcnow().isoformat() + "Z", "confidence": 0.75}
            }

        # 3) Schema coercion + HIL rule
        final_envelope = coerce_schema(gpt_out)

        st.success("✅ ClaimEnvelope generated")
        st.subheader("ClaimEnvelope")
        st.json(final_envelope)

        st.download_button(
            "📥 Download ClaimEnvelope JSON",
            data=json.dumps(final_envelope, indent=2),
            file_name="claim_envelope.json",
            mime="application/json"
        )

# Footer (no secrets, no PHI)
st.markdown("---")
st.caption("Demo only — emits no external events and stores nothing. For production, add RBAC, audits, event bus, and OCR pipeline.")
