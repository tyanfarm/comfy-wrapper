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
from fastapi.responses import StreamingResponse


HOST = os.environ.get("API_SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_SERVER_PORT", "8000"))
COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", "http://localhost:8188")
WORKFLOW_PATH = Path(__file__).with_name("api-workflow.json")

DEFAULT_FPS = 24.0
DEFAULT_DURATION = 15.0

UPLOAD_ENDPOINT = f"{COMFYUI_BASE_URL}/api/upload/image"
PROMPT_ENDPOINT = f"{COMFYUI_BASE_URL}/api/prompt"
QUEUE_ENDPOINT = f"{COMFYUI_BASE_URL}/api/queue"
HISTORY_ENDPOINT = f"{COMFYUI_BASE_URL}/api/history"
VIEW_ENDPOINT = f"{COMFYUI_BASE_URL}/api/view"

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
    return f"{timestamp}_{safe_stem[:10 ]}{suffix}"


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


def get_comfy_json(url: str) -> Any:
    response = requests.get(url, timeout=120)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error": "ComfyUI request failed",
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


def queue_contains_job(items: list[Any], job_id: str) -> bool:
    for item in items:
        if isinstance(item, dict):
            if item.get("prompt_id") == job_id or item.get("job_id") == job_id:
                return True
            if queue_contains_job(list(item.values()), job_id):
                return True
        elif isinstance(item, (list, tuple)):
            if any(element == job_id for element in item):
                return True
            if queue_contains_job(list(item), job_id):
                return True
    return False


def get_history_entry(job_id: str) -> dict[str, Any] | None:
    response = requests.get(f"{HISTORY_ENDPOINT}/{job_id}", timeout=120)
    if response.status_code == 404:
        return None

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error": "ComfyUI request failed",
                "response": response.text,
            },
        ) from exc

    history_data = response.json()
    if isinstance(history_data, dict):
        entry = history_data.get(job_id)
        if isinstance(entry, dict):
            return entry
    return None


def get_job_status_payload(job_id: str) -> dict[str, Any]:
    history_entry = get_history_entry(job_id)
    if history_entry is not None:
        status_info = history_entry.get("status", {})
        status_str = status_info.get("status_str") or "completed"
        video_info = extract_node_341_video(history_entry)
        return {
            "job_id": job_id,
            "status": status_str,
            "done": True,
            "video": video_info,
            "history": history_entry,
        }

    queue_data = get_comfy_json(QUEUE_ENDPOINT)
    running_items = queue_data.get("queue_running", []) if isinstance(queue_data, dict) else []
    pending_items = queue_data.get("queue_pending", []) if isinstance(queue_data, dict) else []

    if queue_contains_job(running_items, job_id):
        return {"job_id": job_id, "status": "running", "done": False}
    if queue_contains_job(pending_items, job_id):
        return {"job_id": job_id, "status": "pending", "done": False}

    raise HTTPException(status_code=404, detail=f'Job "{job_id}" was not found in ComfyUI queue or history.')


def extract_node_341_video(history_entry: dict[str, Any]) -> dict[str, Any] | None:
    outputs = history_entry.get("outputs", {})
    node_341 = outputs.get("341")
    if not isinstance(node_341, dict):
        return None

    images = node_341.get("images")
    if not isinstance(images, list) or not images:
        return None

    first_video = images[0]
    if not isinstance(first_video, dict):
        return None

    filename = first_video.get("filename")
    file_type = first_video.get("type")
    if not filename or not file_type:
        return None

    subfolder = first_video.get("subfolder", "")
    params = {
        "filename": filename,
        "type": file_type,
        "subfolder": subfolder,
    }
    return {
        "filename": filename,
        "subfolder": subfolder,
        "type": file_type,
        "url": requests.Request("GET", VIEW_ENDPOINT, params=params).prepare().url,
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

    prompt_data = prompt_response.json()
    job_id = prompt_data.get("prompt_id")
    if not job_id:
        raise HTTPException(status_code=502, detail="ComfyUI response missing prompt_id.")

    return {
        "message": "Workflow started",
        "job_id": job_id,
    }


@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str) -> dict[str, Any]:
    status_payload = get_job_status_payload(job_id)
    return {
        "job_id": status_payload["job_id"],
        "status": status_payload["status"],
        "done": status_payload["done"],
    }


@app.get("/jobs/{job_id}/video")
def get_job_video(job_id: str) -> StreamingResponse:
    status_payload = get_job_status_payload(job_id)
    if not status_payload["done"]:
        raise HTTPException(status_code=409, detail=f'Job "{job_id}" is still {status_payload["status"]}.')

    video_info = status_payload.get("video")
    if not isinstance(video_info, dict):
        raise HTTPException(status_code=404, detail=f'Job "{job_id}" does not have node 341 video output yet.')

    response = requests.get(video_info["url"], stream=True, timeout=120)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error": "ComfyUI video fetch failed",
                "response": response.text,
            },
        ) from exc

    media_type = response.headers.get("Content-Type", "video/mp4")
    headers = {"Content-Disposition": f'inline; filename="{video_info["filename"]}"'}
    return StreamingResponse(response.iter_content(chunk_size=8192), media_type=media_type, headers=headers)


if __name__ == "__main__":
    uvicorn.run("api_server:app", host=HOST, port=PORT, reload=False)
