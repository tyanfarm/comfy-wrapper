import json
import mimetypes
import os
import re
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


HOST = os.environ.get("API_SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_SERVER_PORT", "8000"))
COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", "http://localhost:8188")
WORKFLOW_PATH = Path(__file__).with_name("api-workflow.json")

DEFAULT_FPS = 24.0
DEFAULT_DURATION = 14.0

UPLOAD_ENDPOINT = f"{COMFYUI_BASE_URL}/api/upload/image"
PROMPT_ENDPOINT = f"{COMFYUI_BASE_URL}/api/prompt"

app = FastAPI(title="ComfyUI Workflow API")


def load_workflow_template() -> dict[str, Any]:
    with WORKFLOW_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


WORKFLOW_TEMPLATE = load_workflow_template()


def sanitize_filename(filename: str, prefix: str) -> str:
    original_name = Path(filename or "").name
    stem = Path(original_name).stem or prefix
    suffix = Path(original_name).suffix
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{safe_stem}{suffix}"


def parse_numeric_field(raw_value: str | None, field_name: str, default: float) -> float:
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f'Field "{field_name}" must be a number.') from exc


async def read_uploaded_file(file: UploadFile, label: str) -> tuple[str, bytes, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail=f'Missing required file field "{label}".')

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f'"{label}" file is empty.')

    renamed_filename = sanitize_filename(file.filename, label)
    content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
    return renamed_filename, content, content_type


def upload_to_comfy(filename: str, content: bytes, content_type: str) -> dict[str, Any]:
    response = requests.post(
        UPLOAD_ENDPOINT,
        files={"image": (filename, content, content_type)},
        timeout=120,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error": "ComfyUI upload failed",
                "response": response.text,
            },
        ) from exc
    return response.json()


def build_prompt(image_name: str, video_name: str, fps: float, duration: float) -> dict[str, Any]:
    workflow = deepcopy(WORKFLOW_TEMPLATE)
    workflow["269"]["inputs"]["image"] = image_name
    workflow["345"]["inputs"]["video"] = video_name
    workflow["358"]["inputs"]["value"] = fps
    workflow["427"]["inputs"]["value"] = duration

    return {
        "client_id": str(uuid.uuid4()),
        "prompt": workflow,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    video: UploadFile = File(...),
    image: UploadFile = File(...),
    fps: str | None = Form(None),
    durations: str | None = Form(None),
) -> dict[str, Any]:
    resolved_fps = parse_numeric_field(fps, "fps", DEFAULT_FPS)
    resolved_duration = parse_numeric_field(durations, "durations", DEFAULT_DURATION)

    image_name, image_content, image_type = await read_uploaded_file(image, "image")
    video_name, video_content, video_type = await read_uploaded_file(video, "video")

    image_upload = upload_to_comfy(image_name, image_content, image_type)
    video_upload = upload_to_comfy(video_name, video_content, video_type)

    prompt_payload = build_prompt(image_name, video_name, resolved_fps, resolved_duration)
    prompt_response = requests.post(PROMPT_ENDPOINT, json=prompt_payload, timeout=600)
    try:
        prompt_response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=prompt_response.status_code,
            detail={
                "error": "ComfyUI prompt failed",
                "response": prompt_response.text,
            },
        ) from exc

    return {
        "message": "Workflow started",
            "uploaded": {
                "image": image_upload,
                "video": video_upload,
            },
            "workflow_values": {
                "image": image_name,
                "video": video_name,
                "fps": resolved_fps,
                "durations": resolved_duration,
            },
        "prompt_response": prompt_response.json(),
    }


if __name__ == "__main__":
    uvicorn.run("api_server:app", host=HOST, port=PORT, reload=False)
