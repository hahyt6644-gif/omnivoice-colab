# omnivoice_app.py
# Complete merged OmniVoice app: Gradio UI + FastAPI endpoints
# Mount: /ui → Gradio, /api/clone → Voice Clone API, /api/design → Voice Design API

import os
import sys
import logging
import tempfile
import shutil
import subprocess
import uuid
import re
from typing import Any, Dict, Optional

import numpy as np
import torch
import scipy.io.wavfile as wavfile

# ---------------------------------------------------------------------------
# Directory Setup
# ---------------------------------------------------------------------------
temp_audio_dir = "./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Path Setup & Model Imports
# ---------------------------------------------------------------------------
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)

from subtitle import subtitle_maker
try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)
logger = logging.getLogger("omnivoice_app")

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
print("Loading model from k2-fsa/OmniVoice to cuda ...")

try:
    from hf_mirror import download_model
    HF_MIRROR_AVAILABLE = True
except ImportError:
    HF_MIRROR_AVAILABLE = False

try:
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )
except Exception as e:
    if not HF_MIRROR_AVAILABLE:
        raise RuntimeError("Model load failed and hf_mirror is not available.") from e

    omnivoice_model_path = download_model(
        "k2-fsa/OmniVoice",
        download_folder="./OmniVoice_Model",
        redownload=False,
        workers=6,
        use_snapshot=False,
    )
    model = OmniVoice.from_pretrained(
        omnivoice_model_path,
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )

sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}

EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]",
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]",
    "[dissatisfaction-hnn]"
]

_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
    "English Accent": [
        "American Accent", "Australian Accent", "British Accent", "Chinese Accent",
        "Canadian Accent", "Indian Accent", "Korean Accent", "Portuguese Accent",
        "Russian Accent", "Japanese Accent"
    ],
    "Chinese Dialect": [
        "Henan Dialect", "Shaanxi Dialect", "Sichuan Dialect", "Guizhou Dialect",
        "Yunnan Dialect", "Guilin Dialect", "Jinan Dialect", "Shijiazhuang Dialect",
        "Gansu Dialect", "Ningxia Dialect", "Qingdao Dialect", "Northeast Dialect"
    ],
}

DIALECT_MAP = {
    "Henan Dialect": "河南话", "Shaanxi Dialect": "陕西话", "Sichuan Dialect": "四川话",
    "Guizhou Dialect": "贵州话", "Yunnan Dialect": "云南话", "Guilin Dialect": "桂林话",
    "Jinan Dialect": "济南话", "Shijiazhuang Dialect": "石家庄话", "Gansu Dialect": "甘肃话",
    "Ningxia Dialect": "宁夏话", "Qingdao Dialect": "青岛话", "Northeast Dialect": "东北话",
}

_ATTR_INFO = {
    "English Accent": "Only effective for English speech.",
    "Chinese Dialect": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# JS for tag insertion (Gradio UI)
# ---------------------------------------------------------------------------
INSERT_TAG_JS_VC = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vc_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

INSERT_TAG_JS_VD = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vd_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

# ---------------------------------------------------------------------------
# Shared Helper Functions
# ---------------------------------------------------------------------------

def tts_file_name(text: str, language: str = "en") -> str:
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip().replace(" ", "_")
    if not clean_text:
        clean_text = "audio"
    truncated = clean_text[:20]
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"
    rand = uuid.uuid4().hex[:8].upper()
    return f"{temp_audio_dir}/{truncated}_{lang}_{rand}.wav"


def _is_whisper_supported(lang: Optional[str]) -> bool:
    if not lang or lang == "Auto":
        return True
    if WHISPER_LANGUAGE_CODE is None:
        return True
    supported = (
        [str(k).lower() for k in WHISPER_LANGUAGE_CODE.keys()] +
        [str(v).lower() for v in WHISPER_LANGUAGE_CODE.values()]
    )
    lang_lower = lang.lower()
    return any(w in lang_lower or lang_lower in w for w in supported)


def generate_subtitles_if_needed(wav_path: str, lang: Optional[str], want_subs: bool):
    """Returns (sentence_srt, word_srt, shorts_srt) or (None, None, None)."""
    if not want_subs:
        return None, None, None
    if not _is_whisper_supported(lang):
        logger.warning(f"Language '{lang}' unsupported by Whisper. Skipping subtitles.")
        return None, None, None
    try:
        whisper_lang = lang if (lang and lang != "Auto") else None
        whisper_results = subtitle_maker(wav_path, whisper_lang)
        if whisper_results and len(whisper_results) > 3:
            return whisper_results[1], whisper_results[2], whisper_results[3]
    except Exception as e:
        logger.warning(f"Subtitle generation failed: {e}")
    return None, None, None


def convert_to_wav(input_path: str) -> str:
    """
    Convert any supported audio format to WAV using ffmpeg.
    Returns path to the converted WAV file (saved in temp_audio_dir).
    Raises RuntimeError if ffmpeg fails.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path  # Already WAV, no conversion needed

    out_path = os.path.join(temp_audio_dir, f"converted_{uuid.uuid4().hex[:8]}.wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", out_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
    return out_path


def _gen_core(
    text: str,
    language: Optional[str],
    ref_audio: Optional[str],
    instruct: Optional[str],
    num_step: int,
    guidance_scale: float,
    denoise: bool,
    speed: float,
    duration: Optional[float],
    preprocess_prompt: bool,
    postprocess_output: bool,
    mode: str,
    ref_text: Optional[str] = None,
):
    """
    Core TTS generation. Returns ((sr, waveform_int16), status_str) or (None, error_str).
    """
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    # Auto-transcribe ref audio if not provided for clone mode
    if mode == "clone" and ref_audio and not ref_text:
        try:
            whisper_lang = language if (language and language != "Auto") else None
            whisper_results = subtitle_maker(ref_audio, whisper_lang)
            if whisper_results and len(whisper_results) > 7:
                ref_text = whisper_results[7]
        except Exception as e:
            logger.warning(f"Fallback transcription failed: {e}")

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text
        )
    elif mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "Done."


# ---------------------------------------------------------------------------
# FastAPI Setup
# ---------------------------------------------------------------------------
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import httpx
from starlette.requests import Request
api_app = FastAPI(title="OmniVoice API", version="1.0.0")

# Serve generated audio files at /audio/<filename>
api_app.mount("/audio", StaticFiles(directory=temp_audio_dir), name="audio")


def _wav_url_from_path(request_base_url: str, wav_path: str) -> str:
    """Build a publicly accessible URL for a generated WAV file."""
    filename = os.path.basename(wav_path)
    base = str(request_base_url).rstrip("/")
    return f"{base}/audio/{filename}"


async def _download_audio_from_url(url: str) -> str:
    """Download audio from a URL, save to temp dir, return local path."""
    ext = os.path.splitext(url.split("?")[0])[-1].lower()
    if ext not in SUPPORTED_AUDIO_FORMATS:
        ext = ".mp3"  # Fallback for URLs without clear extension
    tmp_path = os.path.join(temp_audio_dir, f"dl_{uuid.uuid4().hex[:8]}{ext}")
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {e}")
    return tmp_path


def _build_srt_content(srt_path: Optional[str]) -> Optional[str]:
    """Read SRT file content as string, return None if not available."""
    if not srt_path or not os.path.isfile(srt_path):
        return None
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _cleanup(*paths):
    """Safely delete temporary files."""
    for p in paths:
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except Exception as e:
                logger.warning(f"Cleanup failed for {p}: {e}")


# ---------------------------------------------------------------------------
# /api/clone  — Voice Clone Endpoint
# ---------------------------------------------------------------------------
@api_app.post("/api/clone")
async def api_clone(
    request: Request,
    text: str = Form(...),
    language: Optional[str] = Form("Auto"),
    ref_text: Optional[str] = Form(None),
    ref_audio_url: Optional[str] = Form(None),
    want_subs: bool = Form(False),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    denoise: bool = Form(True),
    speed: float = Form(1.0),
    duration: Optional[float] = Form(None),
    preprocess_prompt: bool = Form(True),
    postprocess_output: bool = Form(True),
    ref_audio: Optional[UploadFile] = File(None),
):
    """
    Voice Clone endpoint.

    Send either:
      - ref_audio (file upload), OR
      - ref_audio_url (URL string)

    Supports: .wav .mp3 .m4a .aac .ogg .flac .webm (auto-converted to WAV via ffmpeg)
    """
    tmp_uploaded = None
    tmp_converted = None

    try:
        # ── 1. Resolve reference audio ──────────────────────────────────────
        if ref_audio is not None and ref_audio.filename:
            ext = os.path.splitext(ref_audio.filename)[-1].lower()
            if ext not in SUPPORTED_AUDIO_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported audio format '{ext}'. Supported: {SUPPORTED_AUDIO_FORMATS}"
                )
            tmp_uploaded = os.path.join(temp_audio_dir, f"up_{uuid.uuid4().hex[:8]}{ext}")
            with open(tmp_uploaded, "wb") as f:
                shutil.copyfileobj(ref_audio.file, f)
            ref_audio_path = tmp_uploaded

        elif ref_audio_url:
            ref_audio_path = await _download_audio_from_url(ref_audio_url)
            tmp_uploaded = ref_audio_path  # track for cleanup

        else:
            raise HTTPException(status_code=400, detail="Provide either 'ref_audio' or 'ref_audio_url'.")

        # ── 2. Convert to WAV if needed ─────────────────────────────────────
        try:
            wav_ref_path = convert_to_wav(ref_audio_path)
            if wav_ref_path != ref_audio_path:
                tmp_converted = wav_ref_path
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        # ── 3. Generate TTS ──────────────────────────────────────────────────
        result, status = _gen_core(
            text=text,
            language=language,
            ref_audio=wav_ref_path,
            instruct=None,
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            speed=speed,
            duration=duration,
            preprocess_prompt=preprocess_prompt,
            postprocess_output=postprocess_output,
            mode="clone",
            ref_text=ref_text or None,
        )

        if result is None:
            raise HTTPException(status_code=500, detail=status)

        sr, waveform = result
        out_wav = tts_file_name(text, language=language or "auto")
        wavfile.write(out_wav, sr, waveform)

        # ── 4. Subtitles ─────────────────────────────────────────────────────
        c_srt, w_srt, s_srt = generate_subtitles_if_needed(out_wav, language, want_subs)

        # ── 5. Build response ────────────────────────────────────────────────
        base_url = str(request.base_url).rstrip("/")
        audio_filename = os.path.basename(out_wav)
        audio_url = f"{base_url}/audio/{audio_filename}"

        return JSONResponse({
            "success": True,
            "audio_url": audio_url,
            "sentence_srt": _build_srt_content(c_srt),
            "word_srt": _build_srt_content(w_srt),
            "shorts_srt": _build_srt_content(s_srt),
            "status": status,
        })

    finally:
        # Cleanup temp uploaded/downloaded files (keep the output WAV)
        _cleanup(tmp_uploaded if tmp_uploaded != wav_ref_path else None)
        _cleanup(tmp_converted)


# ---------------------------------------------------------------------------
# /api/design  — Voice Design Endpoint
# ---------------------------------------------------------------------------
@api_app.post("/api/design")
async def api_design(
    request: Request,
    text: str = Form(...),
    language: Optional[str] = Form("Auto"),
    want_subs: bool = Form(False),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    denoise: bool = Form(True),
    speed: float = Form(1.0),
    duration: Optional[float] = Form(None),
    preprocess_prompt: bool = Form(True),
    postprocess_output: bool = Form(True),
    # Voice design attributes (all optional)
    gender: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    pitch: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
    english_accent: Optional[str] = Form(None),
    chinese_dialect: Optional[str] = Form(None),
):
    """
    Voice Design endpoint.

    Optional voice attributes:
      gender, age, pitch, style, english_accent, chinese_dialect
    """
    # Build instruct string from provided attributes
    attrs = [gender, age, pitch, style, english_accent, chinese_dialect]
    selected = [a for a in attrs if a and a.lower() not in ("auto", "none", "")]
    instruct = ", ".join([DIALECT_MAP.get(v, v) for v in selected]) if selected else None

    result, status = _gen_core(
        text=text,
        language=language,
        ref_audio=None,
        instruct=instruct,
        num_step=num_step,
        guidance_scale=guidance_scale,
        denoise=denoise,
        speed=speed,
        duration=duration,
        preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
        mode="design",
    )

    if result is None:
        raise HTTPException(status_code=500, detail=status)

    sr, waveform = result
    out_wav = tts_file_name(text, language=language or "auto")
    wavfile.write(out_wav, sr, waveform)

    c_srt, w_srt, s_srt = generate_subtitles_if_needed(out_wav, language, want_subs)

    base_url = str(request.base_url).rstrip("/")
    audio_filename = os.path.basename(out_wav)
    audio_url = f"{base_url}/audio/{audio_filename}"

    return JSONResponse({
        "success": True,
        "audio_url": audio_url,
        "sentence_srt": _build_srt_content(c_srt),
        "word_srt": _build_srt_content(w_srt),
        "shorts_srt": _build_srt_content(s_srt),
        "status": status,
    })


# ---------------------------------------------------------------------------
# Gradio UI (unchanged from original)
# ---------------------------------------------------------------------------
import gradio as gr

theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.1em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}

.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

def _lang_dropdown(label="Language (optional)", value="Auto"):
    return gr.Dropdown(
        label=label, choices=_ALL_LANGUAGES, value=value,
        allow_custom_value=False, interactive=True,
    )

def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed",
                       info="1.0 = normal. >1 faster, <1 slower.")
        du = gr.Number(value=None, label="Duration (seconds)",
                       info="Set a fixed duration to override speed.")
        ns = gr.Slider(4, 64, value=32, step=1, label="Inference Steps",
                       info="Lower = faster, higher = better quality.")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
        pp = gr.Checkbox(label="Preprocess Prompt", value=True,
                         info="Applies silence removal and trims reference audio.")
        po = gr.Checkbox(label="Postprocess Output", value=True,
                         info="Removes long silences from generated audio.")
    return ns, gs, dn, sp, du, pp, po

with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as gradio_app:
    gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ OmniVoice Multilingual</h1>
            <p>State-of-the-art text-to-speech model for 600+ languages,
               supporting Voice Clone and Voice Design.</p>
        </div>
    """)

    with gr.Tabs():
        # ── Voice Clone Tab ────────────────────────────────────────────────
        with gr.TabItem("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(
                        label="Text to Synthesize", lines=4,
                        placeholder="Enter the text to synthesize...", elem_id="vc_textbox"
                    )

                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vc_text],
                                outputs=vc_text,
                                js=INSERT_TAG_JS_VC
                            )

                    with gr.Row():
                        vc_lang = _lang_dropdown("Language (optional)")
                        vc_want_subs = gr.Checkbox(label="Want Subtitles?", value=False)

                    vc_ref_audio = gr.Audio(
                        label="Reference Audio (3–10 seconds audio)",
                        type="filepath", elem_classes="compact-audio"
                    )
                    vc_ref_text = gr.Textbox(
                        label="Reference Text", lines=2,
                        placeholder="Auto-transcribed upon audio upload. Edit if Whisper gets it wrong."
                    )
                    vc_btn = gr.Button("Generate", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()

                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="Output Audio", type="numpy")
                    vc_status = gr.Textbox(label="Status", lines=1)
                    with gr.Accordion("Download files", open=False):
                        vc_out_wav = gr.File(label="Generated Audio (WAV)")
                        vc_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vc_out_word_srt = gr.File(label="Word Level SRT")
                        vc_out_shorts_srt = gr.File(label="Shorts SRT")

            def _auto_transcribe(audio_path, lang):
                if not audio_path:
                    return gr.update(value="")
                try:
                    whisper_lang = lang if lang != "Auto" else None
                    whisper_results = subtitle_maker(audio_path, whisper_lang)
                    if whisper_results and len(whisper_results) > 7:
                        return gr.update(value=whisper_results[7])
                except Exception as e:
                    logger.warning(f"Auto-transcription failed: {e}")
                return gr.update(value="")

            vc_ref_audio.change(
                fn=_auto_transcribe,
                inputs=[vc_ref_audio, vc_lang],
                outputs=[vc_ref_text]
            )

            def _clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po):
                res = _gen_core(
                    text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po,
                    mode="clone", ref_text=ref_text
                )
                if res[0] is None:
                    return None, res[1], None, None, None, None

                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vc_btn.click(
                _clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs,
                        vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                outputs=[vc_audio, vc_status, vc_out_wav,
                         vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt],
            )

        # ── Voice Design Tab ───────────────────────────────────────────────
        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(
                        label="Text to Synthesize", lines=4,
                        placeholder="Enter the text to synthesize...", elem_id="vd_textbox"
                    )

                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vd_text],
                                outputs=vd_text,
                                js=INSERT_TAG_JS_VD
                            )

                    with gr.Row():
                        vd_lang = _lang_dropdown(value='Auto')
                        vd_want_subs = gr.Checkbox(label="Want Subtitles?", value=False)

                    vd_btn = gr.Button("Generate", variant="primary")

                    with gr.Accordion("Character Voice Design", open=False):
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            default_val = "Auto"
                            if _cat == "Gender":
                                default_val = "Female"
                            elif _cat == "Age":
                                default_val = "Young Adult"
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=["Auto"] + _choices,
                                    value=default_val,
                                    info=_ATTR_INFO.get(_cat)
                                )
                            )

                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()

                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="Output Audio", type="numpy")
                    vd_status = gr.Textbox(label="Status", lines=1)
                    with gr.Accordion("Download files", open=False):
                        vd_out_wav = gr.File(label="Generated Audio (WAV)")
                        vd_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vd_out_word_srt = gr.File(label="Word Level SRT")
                        vd_out_shorts_srt = gr.File(label="Shorts SRT")

            def _build_instruct(groups):
                selected = [g for g in groups if g and g != "Auto"]
                if not selected:
                    return None
                return ", ".join([DIALECT_MAP.get(v, v) for v in selected])

            def _design_fn(text, lang, want_subs, ns, gs, dn, sp, du, pp, po, *groups):
                instruct = _build_instruct(groups)
                res = _gen_core(
                    text, lang, None, instruct, ns, gs, dn, sp, du, pp, po,
                    mode="design"
                )
                if res[0] is None:
                    return None, res[1], None, None, None, None

                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vd_btn.click(
                _design_fn,
                inputs=[vd_text, vd_lang, vd_want_subs,
                        vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                outputs=[vd_audio, vd_status, vd_out_wav,
                         vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt],
            )

# ---------------------------------------------------------------------------
# Mount Gradio on FastAPI at /ui
# ---------------------------------------------------------------------------
gradio_app.queue()  # Enable Gradio queue support

# Mount Gradio ASGI app at /ui
from gradio import mount_gradio_app
app = mount_gradio_app(api_app, gradio_app, path="/ui")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
