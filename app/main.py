import os
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

app = FastAPI(title="AI Judge Pipeline")


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def upload_pdf_to_openai(file: UploadFile) -> str:
    """
    Uploads a PDF to OpenAI Files API and returns file_id.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type} ({file.filename})")

    try:
        client = get_openai_client()
        # Upload as a file usable by Responses API
        uploaded = client.files.create(
            file=(file.filename, file.file, "application/pdf"),
            purpose="user_data",
        )
        return uploaded.id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF to OpenAI: {e}")


def grounded_fact_summary_pdf(file_ids: List[str], extra: Optional[str] = None) -> str:
    """
    Step 1: Extract verifiable facts only, with evidence quotes.
    Input is one or more PDFs (file_ids).
    Returns text (structured summary).
    """
    prompt = (
        "You are extracting verifiable facts ONLY from the provided case documents.\n\n"
        "Rules:\n"
        "- Use ONLY information explicitly stated in the documents.\n"
        "- Do NOT infer missing details.\n"
        "- For each key fact, include a short evidence quote (max 25 words) copied from the documents.\n"
        "- If a fact is unclear or disputed, mark it as 'unknown' or 'disputed'.\n"
        "- Do NOT merge separate incidents, parties, or dates unless explicitly linked in the text.\n"
        "- Keep it structured and concise.\n\n"
        "Return sections:\n"
        "1) Parties\n"
        "2) Timeline (dates/sequence)\n"
        "3) Alleged acts / charges (as stated)\n"
        "4) Evidence mentioned\n"
        "5) Aggravating factors (explicit)\n"
        "6) Mitigating factors (explicit)\n"
        "7) Contradictions / disputed points\n"
        "8) Unknowns / gaps\n"
        "9) Potentially outcome-determinative facts (only explicit in text)\n"
    )

    if extra and extra.strip():
        prompt += f"\nExtra instructions:\n{extra.strip()}\n"

    content: List[Dict[str, Any]] = []
    for fid in file_ids:
        content.append({"type": "input_file", "file_id": fid})
    content.append({"type": "input_text", "text": prompt})

    client = get_openai_client()
    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "low"},
        store=False,
    )
    return resp.output_text


def omission_guard_review_pdf(file_ids: List[str], facts_summary: str, extra: Optional[str] = None) -> str:
    """
    Step 1.5: Detect potentially vital facts that may be missing, underweighted,
    or contradictory relative to the grounded summary.
    Returns text addendum used by the final judge step.
    """
    prompt = (
        "You are a legal QA reviewer checking whether a case summary missed crucial details.\n\n"
        "Inputs: the original case documents + an existing facts summary.\n\n"
        "Rules:\n"
        "- Use ONLY document text.\n"
        "- Do NOT invent any legal rules or facts.\n"
        "- Focus on omissions that could materially affect charging, culpability, or sentencing.\n"
        "- For each finding, include one short direct quote (max 25 words).\n"
        "- If no material omission is found, explicitly write 'No material omissions found'.\n\n"
        "Return sections:\n"
        "1) Potentially omitted material facts\n"
        "2) Underemphasized facts (present but likely underweighted)\n"
        "3) Contradictions requiring resolution\n"
        "4) Ambiguities that could change sentencing\n"
        "5) Minimal additional information needed for reliable sentencing\n"
    )

    if extra and extra.strip():
        prompt += f"\nExtra instructions:\n{extra.strip()}\n"

    user_text = (
        "EXISTING FACTS SUMMARY:\n"
        f"{facts_summary}\n\n"
        "Compare this summary against the original documents and return only evidence-grounded findings.\n"
    )

    content: List[Dict[str, Any]] = []
    for fid in file_ids:
        content.append({"type": "input_file", "file_id": fid})
    content.append({"type": "input_text", "text": user_text + "\n" + prompt})

    client = get_openai_client()
    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "high"},
        store=False,
    )
    return resp.output_text


def judge_decision_pdf(
    file_ids: List[str],
    facts_summary: str,
    omission_review: Optional[str] = None,
    extra: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 2: Produce strict JSON output using JSON Schema structured output.
    Input is PDFs + the grounded facts summary.
    """
    system_instructions = (
        "You are an AI judge.\n\n"
        "You MUST follow these rules:\n"
        "- Judge solely on Swedish laws AND the context given in the case documents.\n"
        "- Do NOT use outside knowledge, memory, or internet sources.\n"
        "- Do NOT invent statutes, legal principles, precedents, facts, or evidence.\n"
        "- Treat the omission/contradiction review as a mandatory safety check before final sentencing.\n"
        "- Re-check your conclusion against the ORIGINAL documents, not only the summary.\n"
        "- If material contradictions or unresolved ambiguities remain, sentencing must be 'unknown'.\n"
        "- If the documents do not provide enough information to justify a sentencing decision, set sentencing to 'unknown'\n"
        "  and list exactly what is missing.\n"
        "- Be professional and neutral.\n"
        "- Output MUST match the JSON schema exactly.\n\n"
        "Task:\n"
        "Using the provided documents and the facts summary, produce:\n"
        "1) Sentencing\n"
        "2) Motivation for that sentencing (explanation/conclusion)\n"
        "3) Short description of the case\n"
    )

    if extra and extra.strip():
        system_instructions += f"\nExtra instructions:\n{extra.strip()}\n"

    json_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sentencing": {"type": "string"},
            "motivation": {"type": "string"},
            "case_description": {"type": "string"},
            "missing_information": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["sentencing", "motivation", "case_description", "missing_information"]
    }

    user_text = (
        "FACTS SUMMARY (grounded with evidence quotes):\n"
        f"{facts_summary}\n\n"
        "OMISSION / CONTRADICTION REVIEW:\n"
        f"{omission_review or 'No omission review provided.'}\n\n"
        "Now decide based on the documents and the facts summary.\n"
        "Remember: do not invent laws or facts.\n"
        "If any omission, contradiction, or ambiguity could materially affect sentencing and is unresolved, use 'unknown'\n"
        "and list that missing/unclear information precisely.\n"
    )

    content: List[Dict[str, Any]] = []
    for fid in file_ids:
        content.append({"type": "input_file", "file_id": fid})
    content.append({"type": "input_text", "text": user_text})

    client = get_openai_client()
    resp = client.responses.create(
        model=MODEL,
        instructions=system_instructions,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "high"},
        text={
            "format": {
                "type": "json_schema",
                "name": "ai_judge_output",
                "strict": True,
                "schema": json_schema
            }
        },
        store=False,
    )

    return json.loads(resp.output_text)


@app.post("/analyze")
async def analyze(
    files: List[UploadFile] = File(..., description="One or more PDF files"),
    extra_instructions: Optional[str] = Form(None, description="Optional extra instructions (length constraints etc.)"),
):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No PDF files provided")

    # 1) Upload PDFs to OpenAI
    file_ids: List[str] = []
    for f in files:
        file_ids.append(upload_pdf_to_openai(f))

    # 2) Step 1: grounded fact extraction
    facts = grounded_fact_summary_pdf(file_ids, extra=extra_instructions)

    # 2.5) Step 1.5: omission/contradiction guardrail review
    omission_review = omission_guard_review_pdf(file_ids, facts_summary=facts, extra=extra_instructions)

    # 3) Step 2: final decision JSON
    decision = judge_decision_pdf(
        file_ids,
        facts_summary=facts,
        omission_review=omission_review,
        extra=extra_instructions,
    )

    response_payload = {
        "facts_summary": facts,
        "omission_review": omission_review,
        "result": decision,
        "model": MODEL,
    }

    output_path = os.path.join(os.path.dirname(__file__), "AI_Judge_Decision.txt")
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(response_payload, ensure_ascii=True, indent=2))

    return response_payload
