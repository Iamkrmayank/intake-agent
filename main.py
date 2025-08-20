# main.py
# Gooclaim Intake Agent (Azure): FHIR + File Uploads → OCR → (optional) LLM → Normalized JSON
# Python 3.10–3.12 recommended

import io, os, json, hashlib, uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# Load env / settings
# =========================
load_dotenv()

AZURE_DI_ENDPOINT = os.getenv("AZURE_DI_ENDPOINT")
AZURE_API_KEY     = os.getenv("AZURE_API_KEY")

AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
GPT_DEPLOYMENT           = os.getenv("GPT_DEPLOYMENT", "gpt-4")

# =========================
# Azure OpenAI client (robust)
# =========================
# Works with openai>=1.40.0 (AzureOpenAI class) OR falls back to generic OpenAI client pointed at Azure endpoint.
aoai = None
_USE_AZ_CLASS = False
try:
    from openai import AzureOpenAI  # preferred if available
    aoai = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    _USE_AZ_CLASS = True
except Exception:
    try:
        from openai import OpenAI  # fallback style
        # Base URL points directly to this deployment; API version is passed by default_query
        aoai = OpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{GPT_DEPLOYMENT}",
            default_query={"api-version": AZURE_OPENAI_API_VERSION},
        )
        _USE_AZ_CLASS = False
    except Exception as e:
        aoai = None  # LLM is optional; pipeline still works with FHIR+OCR only

def aoai_chat_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Call Azure OpenAI Chat Completions with JSON mode. Returns parsed JSON dict.
    If AOAI isn't configured or call fails, we bubble a minimal structure.
    """
    if not aoai:
        return {"_llm_error": "Azure OpenAI client not configured"}
    try:
        resp = aoai.chat.completions.create(
            model=GPT_DEPLOYMENT,              # required for AzureOpenAI; harmless for fallback
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        return {"_llm_error": f"{e.__class__.__name__}: {e}"}

# =========================
# Azure Document Intelligence (new) or Form Recognizer (fallback)
# =========================
DocI_client = None
use_docint = False  # True => azure-ai-documentintelligence; False => azure-ai-formrecognizer

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
    DocI_client = DocumentIntelligenceClient(AZURE_DI_ENDPOINT, AzureKeyCredential(AZURE_API_KEY))
    use_docint = True
except Exception:
    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential as FRKey
        DocI_client = DocumentAnalysisClient(AZURE_DI_ENDPOINT, FRKey(AZURE_API_KEY))
        use_docint = False
    except Exception:
        DocI_client = None  # OCR optional; still accept FHIR-only requests

def ocr_doc(file_bytes: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (plain_text, raw_result_dict). Uses "prebuilt-read".
    """
    if not DocI_client:
        return "", {"warning": "Document Intelligence client not configured"}
    try:
        if use_docint:
            # New DI SDK (1.x)
            # Uses base64Source request body format (SDK wraps it)
            poller = DocI_client.begin_analyze_document(
                model_id="prebuilt-read",
                analyze_request={"base64Source": file_bytes},
                content_type="application/json",
            )
            result = poller.result()
            lines = []
            for page in result.pages or []:
                for line in page.lines or []:
                    lines.append(line.content)
            return "\n".join(lines).strip(), result.to_dict()
        else:
            # FR 3.x fallback
            poller = DocI_client.begin_analyze_document("prebuilt-read", document=file_bytes)
            result = poller.result()
            lines = []
            for page in result.pages:
                for line in page.lines:
                    lines.append(line.content)
            return "\n".join(lines).strip(), {"pages": len(result.pages)}
    except Exception as e:
        return "", {"error": f"OCR failed: {e.__class__.__name__}: {e}"}

# =========================
# Output schema (Pydantic)
# =========================
class DocOut(BaseModel):
    doc_id: str
    type: Optional[str] = None
    blob_url: Optional[str] = None

class PatientOut(BaseModel):
    mrn: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dob: Optional[str] = None
    sex: Optional[str] = None
    payer_id: Optional[str] = None
    payer_name: Optional[str] = None
    member_id: Optional[str] = None
    group_id: Optional[str] = None

class EncounterOut(BaseModel):
    date: Optional[str] = None
    site: Optional[str] = None
    provider_npi: Optional[str] = None
    location_npi: Optional[str] = None

class MetaOut(BaseModel):
    source: Optional[str] = None
    created_at: str
    confidence: float = 0.0
    connector_id: Optional[str] = None
    ingest_mode: Optional[str] = "sync"
    idempotency_key: str

class GooclaimOut(BaseModel):
    claim_id: str
    patient: PatientOut
    encounter: EncounterOut
    documents: List[DocOut]
    clinical_note: Optional[str] = None
    status: str = "PARSING"
    meta: MetaOut

# =========================
# Utilities
# =========================
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def choose(a, b):
    """Prefer deterministic FHIR value a; else fallback to b; else None."""
    return a if a not in (None, "", []) else (b if b not in (None, "", []) else None)

def pick_identifier(identifiers: List[Dict[str, Any]], want: List[str]) -> Optional[str]:
    for sys in want:
        for i in identifiers or []:
            if i.get("system") and sys.lower() in i["system"].lower():
                v = i.get("value")
                if v:
                    return v
    for i in identifiers or []:
        if i.get("value"):
            return i["value"]
    return None

def resource(bundle: Dict, rtype: str) -> Optional[Dict]:
    for entry in (bundle or {}).get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") == rtype:
            return res
    return None

def resources(bundle: Dict, rtype: str) -> List[Dict]:
    out = []
    for entry in (bundle or {}).get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") == rtype:
            out.append(res)
    return out

# =========================
# FHIR parsers
# =========================
def parse_patient(bundle: Dict) -> PatientOut:
    p = resource(bundle, "Patient") or {}
    first, last = None, None
    if p.get("name"):
        n0 = p["name"][0]
        first = (n0.get("given") or [None])[0]
        last = n0.get("family")
    dob = p.get("birthDate")
    sex = (p.get("gender") or None)
    mrn = pick_identifier(p.get("identifier", []), want=["mrn", "medical", "hospital"])

    cov = resource(bundle, "Coverage") or {}
    payer_id = payer_name = None
    if cov.get("payor"):
        org = None
        for ref in cov["payor"]:
            r = ref.get("reference")
            if r and r.startswith("Organization/"):
                for o in resources(bundle, "Organization"):
                    if o.get("id") == r.split("/")[-1]:
                        org = o; break
        if org:
            payer_name = org.get("name")
            payer_id = pick_identifier(org.get("identifier", []), want=["payer", "payer-id", "naic", "tin"])
    member_id = cov.get("subscriberId")

    group_id = None
    if cov.get("grouping"):
        group_id = cov["grouping"].get("group")
    if not group_id:
        group_id = pick_identifier(cov.get("identifier", []), want=["group", "group-number"])

    return PatientOut(
        mrn=mrn, first_name=first, last_name=last, dob=dob, sex=sex,
        payer_id=payer_id, payer_name=payer_name, member_id=member_id, group_id=group_id
    )

def parse_encounter(bundle: Dict) -> EncounterOut:
    e = resource(bundle, "Encounter") or {}
    date = None
    if e.get("period", {}).get("start"):
        date = e["period"]["start"][:10]
    site = e.get("serviceProvider", {}).get("display")

    prac = resource(bundle, "Practitioner") or {}
    provider_npi = pick_identifier(prac.get("identifier", []), want=["npi", "us-npi"])

    loc = resource(bundle, "Location") or {}
    location_npi = pick_identifier(loc.get("identifier", []), want=["npi", "facility", "location"])

    return EncounterOut(date=date, site=site, provider_npi=provider_npi, location_npi=location_npi)

def parse_documents(bundle: Dict) -> List[DocOut]:
    out: List[DocOut] = []
    for i, dr in enumerate(resources(bundle, "DocumentReference")):
        doc_id = dr.get("id") or f"docref-{i+1}"
        dtype = None
        t = dr.get("type", {})
        if t.get("coding"):
            c0 = t["coding"][0]
            dtype = c0.get("display") or c0.get("code")
        blob = None
        if dr.get("content"):
            att = dr["content"][0].get("attachment", {})
            blob = att.get("url")
        out.append(DocOut(doc_id=doc_id, type=dtype, blob_url=blob))
    return out

def parse_clinical_note_from_fhir(bundle: Dict) -> Optional[str]:
    comp = resource(bundle, "Composition") or {}
    if comp.get("section"):
        for s in comp["section"]:
            t = ((s.get("text") or {}).get("div") or "").strip()
            if t:
                title = comp.get("title")
                return f"{title}\n\n{t}" if title else t
    drs = resources(bundle, "DocumentReference")
    if drs:
        att = (drs[0].get("content") or [{}])[0].get("attachment", {})
        if att.get("title"): return att["title"]
    return None

# =========================
# LLM extraction (optional)
# =========================
EXTRACT_SYSTEM = """You extract patient/encounter/payer details from free-text clinical notes.
Return STRICT JSON with keys:
patient: {mrn, first_name, last_name, dob, sex, payer_id, payer_name, member_id, group_id}
encounter: {date, site, provider_npi, location_npi}
clinical_note: string
Rules:
- If a field is unknown, set it to null. Do NOT invent values.
- date/dob: format YYYY-MM-DD if possible, else null.
- sex: one of M/F/O/U if explicit; else null.
- Keep clinical_note as the source text (trim ~6000 chars).
- Output ONLY JSON (no prose).
"""

def llm_extract_from_text(text: str) -> Dict[str, Any]:
    if not text or not aoai:
        return {
            "patient": {k: None for k in ["mrn","first_name","last_name","dob","sex","payer_id","payer_name","member_id","group_id"]},
            "encounter": {k: None for k in ["date","site","provider_npi","location_npi"]},
            "clinical_note": (text[:6000] if text else None)
        }
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": "Extract fields from the following text:\n\n" + text[:12000]},
    ]
    data = aoai_chat_json(messages)
    # normalize keys
    data.setdefault("patient", {})
    data.setdefault("encounter", {})
    data.setdefault("clinical_note", None)
    return data

# =========================
# Merge helpers
# =========================
def merge_patient(fhir_p: PatientOut, llm_p: Dict[str, Any]) -> PatientOut:
    return PatientOut(
        mrn=choose(fhir_p.mrn, llm_p.get("mrn")),
        first_name=choose(fhir_p.first_name, llm_p.get("first_name")),
        last_name=choose(fhir_p.last_name, llm_p.get("last_name")),
        dob=choose(fhir_p.dob, llm_p.get("dob")),
        sex=choose(fhir_p.sex, llm_p.get("sex")),
        payer_id=choose(fhir_p.payer_id, llm_p.get("payer_id")),
        payer_name=choose(fhir_p.payer_name, llm_p.get("payer_name")),
        member_id=choose(fhir_p.member_id, llm_p.get("member_id")),
        group_id=choose(fhir_p.group_id, llm_p.get("group_id")),
    )

def merge_encounter(fhir_e: EncounterOut, llm_e: Dict[str, Any]) -> EncounterOut:
    return EncounterOut(
        date=choose(fhir_e.date, llm_e.get("date")),
        site=choose(fhir_e.site, llm_e.get("site")),
        provider_npi=choose(fhir_e.provider_npi, llm_e.get("provider_npi")),
        location_npi=choose(fhir_e.location_npi, llm_e.get("location_npi")),
    )

def guess_confidence(has_fhir: bool, has_files: bool, llm_ok: bool) -> float:
    base = 0.3
    if has_fhir: base += 0.35
    if has_files: base += 0.25
    if llm_ok: base += 0.1
    return min(0.98, base)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Gooclaim Intake Agent (Azure)", version="1.2.0")

@app.get("/")
def root():
    return {"ok": True, "service": "Gooclaim Intake Agent (Azure)", "version": "1.2.0"}

@app.post("/ingest")
async def ingest(
    files: Optional[List[UploadFile]] = File(default=None),
    fhir_bundle: Optional[str] = Form(default=None),
    json_payload: Optional[str] = Form(default=None),
):
    """
    Accepts multipart/form-data:
      - files: one or more clinician docs (PDF/PNG/JPG/TIFF/TXT)
      - fhir_bundle: JSON string (FHIR Bundle)
      - json_payload: JSON string (e.g., claim_id, connector_id, ingest_mode)
    Returns normalized Gooclaim JSON.
    """
    # ---- Parse inputs
    bundle = None
    if fhir_bundle:
        try:
            bundle = json.loads(fhir_bundle)
            if not isinstance(bundle, dict):
                raise ValueError("fhir_bundle must be a JSON object")
        except Exception as e:
            raise HTTPException(400, f"Invalid fhir_bundle: {e}")

    extra = {}
    if json_payload:
        try:
            extra = json.loads(json_payload)
            if not isinstance(extra, dict):
                raise ValueError("json_payload must be a JSON object")
        except Exception as e:
            raise HTTPException(400, f"Invalid json_payload: {e}")

    # ---- FHIR extraction (deterministic)
    fhir_patient   = parse_patient(bundle) if bundle else PatientOut()
    fhir_encounter = parse_encounter(bundle) if bundle else EncounterOut()
    fhir_docs      = parse_documents(bundle) if bundle else []
    fhir_note      = parse_clinical_note_from_fhir(bundle) if bundle else None

    # ---- File OCR + aggregation
    uploaded_docs: List[DocOut] = []
    texts = []
    if files:
        for f in files:
            content = await f.read()
            await f.seek(0)
            digest = sha1(content)
            doc_id = f"up-{digest[:8]}"

            # TODO: Replace with real upload to S3/Blob and use the resulting URL:
            blob_url = f"file://{f.filename}"

            ext = (f.filename or "").split(".")[-1].lower()
            dtype = "clinical-note" if ext in ("txt", "md") else "attachment"
            uploaded_docs.append(DocOut(doc_id=doc_id, type=dtype, blob_url=blob_url))

            text, _raw = ocr_doc(content, f.filename or "file")
            if not text and ext == "txt":
                try:
                    text = content.decode(errors="ignore")
                except Exception:
                    text = ""
            if text:
                texts.append(text)

    combined_text = ("\n\n---\n\n".join(texts)).strip()
    llm_extract = llm_extract_from_text(combined_text) if combined_text else {
        "patient": {}, "encounter": {}, "clinical_note": None
    }

    # ---- Merge: FHIR > LLM > None
    patient   = merge_patient(fhir_patient, llm_extract.get("patient", {}))
    encounter = merge_encounter(fhir_encounter, llm_extract.get("encounter", {}))
    clinical_note = fhir_note or llm_extract.get("clinical_note")

    # ---- Consolidate docs
    documents = fhir_docs + uploaded_docs

    # ---- Meta/IDs
    created_at = now_iso()
    idem_seed = (json.dumps(bundle, sort_keys=True) if bundle else "") + created_at + "".join(d.doc_id for d in documents)
    idempotency_key = sha1(idem_seed.encode())[:20]
    claim_id = extra.get("claim_id") or f"GC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    meta = MetaOut(
        source=("fhir_bundle" if bundle else "file_upload"),
        created_at=created_at,
        confidence=guess_confidence(has_fhir=bool(bundle), has_files=bool(files), llm_ok=("_llm_error" not in llm_extract)),
        connector_id=extra.get("connector_id"),
        ingest_mode=extra.get("ingest_mode", "sync"),
        idempotency_key=idempotency_key
    )

    out = GooclaimOut(
        claim_id=claim_id,
        patient=patient,
        encounter=encounter,
        documents=documents,
        clinical_note=clinical_note,
        status="PARSING",
        meta=meta
    )
    return JSONResponse(out.model_dump())
