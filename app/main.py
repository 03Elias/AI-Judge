import os
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = "gpt-5.2"

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
        # upload as a file usable by Responses API
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
        "INCLUDE:\n"
        "- What happened (proven facts + disputed facts)\n"
        "- What crime the court found / what culpability findings are explicit (e.g., 'uppsåt ej styrkt', 'grovt oaktsam')\n"
        "- Aggravating/mitigating facts\n\n"
        "EXCLUDE (skip completely):\n"
        "- Any punishment type/length (fängelse, villkorlig, skyddstillsyn)\n"
        "- Any sentencing evaluation language (påföljd, straffvärde, straffmätning, avräkning)\n"
        "- Any outcomes about damages (skadestånd, ersättning, avgift, kr)\n"
        "- Appellate outcome confirmations ('fastställer', 'tingsrättens dom ska därför inte ändras')\n"
        "- 'Domslut' sections\n\n"
        "Return sections:\n"
        "1) Parties\n"
        "2) Timeline (dates/sequence)\n"
        "3) What happened (proven facts)\n"
        "4) What happened (disputed facts)\n"
        "5) Alleged acts / charges (as stated)\n"
        "6) Crime/culpability findings (e.g., uppsåt ej styrkt, grovt oaktsam)\n"
        "7) Evidence mentioned\n"
        "8) Aggravating factors (explicit)\n"
        "9) Mitigating factors (explicit)\n"
        "10) Contradictions / disputed points\n"
        "11) Unknowns / gaps\n"
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
    Step 1.5: Refine the Step 1 summary using original PDFs as ground truth.
    Returns a revised grounded summary to be used directly by Step 2.
    """
    prompt = (
        "You are a legal QA reviewer refining a case summary using the original documents.\n\n"
        "Inputs: the original case documents + an existing facts summary.\n\n"
        "Rules:\n"
        "- Use ONLY document text.\n"
        "- Do NOT invent any legal rules or facts.\n"
        "- Add missing material facts, correct inaccuracies, and remove unsupported claims.\n"
        "- Keep evidence-grounded wording only; if uncertain, mark as 'unknown' or 'disputed'.\n"
        "- Keep one short evidence quote (max 25 words) for each key fact.\n\n"
        "INCLUDE:\n"
        "- What happened (proven facts + disputed facts)\n"
        "- What crime the court found / what culpability findings are explicit (e.g., 'uppsåt ej styrkt', 'grovt oaktsam')\n"
        "- Aggravating/mitigating facts\n\n"
        "EXCLUDE (skip completely):\n"
        "- Any punishment type/length (fängelse, villkorlig, skyddstillsyn)\n"
        "- Any sentencing evaluation language (påföljd, straffvärde, straffmätning, avräkning)\n"
        "- Any outcomes about damages (skadestånd, ersättning, avgift, kr)\n"
        "- Appellate outcome confirmations ('fastställer', 'tingsrättens dom ska därför inte ändras')\n"
        "- 'Domslut' sections\n\n"
        "Output requirement:\n"
        "- Return ONLY the revised final summary (not a critique list).\n"
        "- Use these sections exactly:\n"
        "1) Parties\n"
        "2) Timeline (dates/sequence)\n"
        "3) What happened (proven facts)\n"
        "4) What happened (disputed facts)\n"
        "5) Alleged acts / charges (as stated)\n"
        "6) Crime/culpability findings (e.g., uppsåt ej styrkt, grovt oaktsam)\n"
        "7) Evidence mentioned\n"
        "8) Aggravating factors (explicit)\n"
        "9) Mitigating factors (explicit)\n"
        "10) Contradictions / disputed points\n"
        "11) Unknowns / gaps\n"
        "12) Changes from Step 1 summary (added / removed / corrected)\n"
    )

    if extra and extra.strip():
        prompt += f"\nExtra instructions:\n{extra.strip()}\n"

    user_text = (
        "EXISTING FACTS SUMMARY TO REFINE:\n"
        f"{facts_summary}\n\n"
        "Compare this summary against the original documents and return the corrected, improved summary.\n"
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
    revised_summary: str,
    extra: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 2: Produce strict JSON output using JSON Schema structured output.
    Input is only the revised summary from Step 1.5.
    """
    system_instructions = (
        "You are an AI judge.\n\n"
        "You MUST follow these rules:\n"
        "- Judge solely on Swedish laws AND the context given in the revised summary text.\n"
        "- Use general Swedish law knowledge (statutes, legal principles, precedents, sentencing frameworks).\n"
        "- Do NOT use outside case facts beyond the revised summary.\n"
        "- Do NOT invent facts or evidence.\n"
        "- Make a best-effort, evidence-grounded educated guess when certainty is below 100%.\n"
        "- Always state uncertainty clearly in motivation when facts are incomplete, ambiguous, or disputed.\n"
        "- Use 'unknown' only when the evidence is too conflicting or too incomplete for any responsible sentencing estimate.\n"
        "- If certainty is below 100%, explain why and list exactly what additional information would most improve certainty.\n"
        "- You MUST set certainty_level as one of: high, medium, low.\n"
        "- Use high when uncertainty is minimal, medium when material uncertainty exists but a responsible estimate is possible, and low when uncertainty is substantial.\n"
        "- Be professional and neutral.\n"
        "- Output MUST match the JSON schema exactly.\n\n"
        "IMPORTANT - Your input summary contains ONLY:\n"
        "- What happened (proven facts + disputed facts)\n"
        "- What crime the court found / culpability findings\n"
        "- Aggravating/mitigating facts\n"
        "It contains ZERO punishment, sentencing language, damages, or appellate outcomes.\n\n"
        "Task:\n"
        "Using the revised summary, produce:\n"
        "1) Sentencing\n"
        "2) Motivation for that sentencing (explanation/conclusion)\n"
        "3) Short description of the case\n"
        "4) Certainty level (high/medium/low)\n"
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
            "certainty_level": {
                "type": "string",
                "enum": ["high", "medium", "low"]
            },
            "missing_information": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["sentencing", "motivation", "case_description", "certainty_level", "missing_information"]
    }

    user_text = (
        "REVISED SUMMARY (from Step 1.5):\n"
        f"{revised_summary}\n\n"
        "Now decide based only on this revised summary.\n"
        "Remember: do not invent laws or facts.\n"
        "If you are not 100% certain, still provide the best evidence-grounded sentencing estimate and clearly state uncertainty.\n"
        "Use 'unknown' only if no responsible estimate can be made from the revised summary.\n"
        "Always set certainty_level to exactly one of: high, medium, low.\n"
        "Always list missing/unclear information precisely.\n"
    )

    client = get_openai_client()
    resp = client.responses.create(
        model=MODEL,
        instructions=system_instructions,
        input=user_text,
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

    # 2.5) Step 1.5: refine summary using PDFs + Step 1 summary
    revised_summary = omission_guard_review_pdf(file_ids, facts_summary=facts, extra=extra_instructions)

    # 3) Step 2: final decision JSON (input is only Step 1.5 revised summary)
    decision = judge_decision_pdf(
        revised_summary=revised_summary,
        extra=extra_instructions,
    )

    response_payload = {
        "facts_summary": facts,
        "revised_summary": revised_summary,
        "result": decision,
        "model": MODEL,
    }

    output_path = os.path.join(os.path.dirname(__file__), "AI_Judge_Decision.txt")
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(response_payload, ensure_ascii=True, indent=2))

    return response_payload
