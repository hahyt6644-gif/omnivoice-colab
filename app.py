import os
import sys
import logging
import uuid
import re
import threading
import tempfile
import requests
import subprocess
import time
from typing import Any, Dict, Optional

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile

# --- API Modules ---
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import nest_asyncio

# Apply nest_asyncio so Uvicorn can run inside Colab without blocking
nest_asyncio.apply()

# Define directory for temp audio
temp_audio_dir = "./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup path to import subtitle_maker from OmniVoice directory
# ---------------------------------------------------------------------------
OmniVoice_path = os.path.abspath("./OmniVoice/")
sys.path.append(OmniVoice_path)

try:
    from subtitle import subtitle_maker
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    subtitle_maker = None
    WHISPER_LANGUAGE_CODE = None
    logging.warning("Subtitle module could not be loaded. Subtitle features will be disabled.")

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("omnivoice")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Model Loading (Global Scope)
# ---------------------------------------------------------------------------
logger.info("Loading model from k2-fsa/OmniVoice to cuda...")

from hf_mirror import download_model

try:
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )
except Exception as e:
    logger.info("Direct loading failed. Downloader fallback initiated...")
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
logger.info("Model loaded successfully!")

# Thread Lock to safely queue incoming concurrent GPU inference requests
gpu_lock = threading.Lock()

# ---------------------------------------------------------------------------
# FastAPI Setup & Schemas
# ---------------------------------------------------------------------------
app = FastAPI(title="OmniVoice API", description="Production API and UI for OmniVoice Multilingual TTS")

# Mount the audio directory to serve generated WAV files over HTTP
app.mount("/audio", StaticFiles(directory=temp_audio_dir), name="audio")

class VoiceCloneRequest(BaseModel):
    text: str
    language: str = "Auto"
    ref_audio_path: str
    ref_text: Optional[str] = None
    want_subs: bool = False
    num_step: int = 32
    guidance_scale: float = 2.0
    denoise: bool = True
    speed: float = 1.0
    duration: Optional[float] = None
    preprocess_prompt: bool = True
    postprocess_output: bool = True

class VoiceDesignRequest(BaseModel):
    text: str
    language: str = "Auto"
    instruct: Optional[str] = None
    want_subs: bool = False
    num_step: int = 32
    guidance_scale: float = 2.0
    denoise: bool = True
    speed: float = 1.0
    duration: Optional[float] = None
    preprocess_prompt: bool = True
    postprocess_output: bool = True

# ---------------------------------------------------------------------------
# Core Logic Helpers
# ---------------------------------------------------------------------------
def _is_whisper_supported(lang):
    if not lang or lang == "Auto" or WHISPER_LANGUAGE_CODE is None:
        return True 
    supported_langs = [str(k).lower() for k in WHISPER_LANGUAGE_CODE.keys()] + \
                      [str(v).lower() for v in WHISPER_LANGUAGE_CODE.values()]
    lang_lower = lang.lower()
    return any(w_lang in lang_lower or lang_lower in w_lang for w_lang in supported_langs)

def generate_subtitles_if_needed(wav_path, lang, want_subs):
    if not want_subs or subtitle_maker is None:
        return None, None, None
    if not _is_whisper_supported(lang):
        logger.warning(f"Language '{lang}' is unsupported by Whisper. Skipping subtitles.")
        return None, None, None
    try:
        whisper_lang = lang if (lang and lang != "Auto") else None
        whisper_results = subtitle_maker(wav_path, whisper_lang)
        if whisper_results and len(whisper_results) > 3:
            return whisper_results[1], whisper_results[2], whisper_results[3] 
    except Exception as e:
        logger.warning(f"Subtitle generation failed: {e}")
    return None, None, None

def tts_file_name(text, language="en"):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    clean_text = clean_text.lower().strip().replace(" ", "_")
    if not clean_text:
        clean_text = "audio"
    truncated = clean_text[:20]
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"
    rand = uuid.uuid4().hex[:8].upper()
    return f"{temp_audio_dir}/{truncated}_{lang}_{rand}.wav"

def _gen_core(
    text, language, ref_audio, instruct, num_step, guidance_scale, 
    denoise, speed, duration, preprocess_prompt, postprocess_output, mode, ref_text=None
):
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    if mode == "clone" and ref_audio and not ref_text and subtitle_maker is not None:
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
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if mode == "design" and instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    try:
        with gpu_lock:
            audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "Done."

# ---------------------------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

@app.post("/api/generate/clone")
def api_generate_clone(req: VoiceCloneRequest):
    ref_path = req.ref_audio_path
    temp_download_path = None
    
    # 1. Detect if the path is a web URL and download it temporarily
    if ref_path.startswith("http://") or ref_path.startswith("https://"):
        try:
            logger.info(f"Downloading reference audio from URL: {ref_path}")
            r = requests.get(ref_path, stream=True)
            r.raise_for_status()
            
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            for chunk in r.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file.close()
            
            ref_path = tmp_file.name
            temp_download_path = ref_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {e}")
            
    # 2. Otherwise, check if it's a valid local path
    elif not os.path.exists(ref_path):
        raise HTTPException(status_code=400, detail="Reference audio path does not exist on the server.")

    # 3. Generate Audio
    res = _gen_core(
        req.text, req.language, ref_path, None, req.num_step, req.guidance_scale, 
        req.denoise, req.speed, req.duration, req.preprocess_prompt, req.postprocess_output, 
        mode="clone", ref_text=req.ref_text
    )
    
    # 4. Clean up the downloaded temporary file
    if temp_download_path and os.path.exists(temp_download_path):
        os.remove(temp_download_path)
    
    if res[0] is None:
        raise HTTPException(status_code=500, detail=res[1])
        
    # 5. Save and return output
    sr, waveform = res[0]
    tmp_wav = tts_file_name(req.text, language=req.language)
    wavfile.write(tmp_wav, sr, waveform)
    
    generate_subtitles_if_needed(tmp_wav, req.language, req.want_subs)
    filename = os.path.basename(tmp_wav)
    
    return {
        "status": "success",
        "audio_url": f"/audio/{filename}",
        "subtitles_generated": req.want_subs
    }

@app.post("/api/generate/design")
def api_generate_design(req: VoiceDesignRequest):
    res = _gen_core(
        req.text, req.language, None, req.instruct, req.num_step, req.guidance_scale, 
        req.denoise, req.speed, req.duration, req.preprocess_prompt, req.postprocess_output, 
        mode="design"
    )
    
    if res[0] is None:
        raise HTTPException(status_code=500, detail=res[1])
        
    sr, waveform = res[0]
    tmp_wav = tts_file_name(req.text, language=req.language)
    wavfile.write(tmp_wav, sr, waveform)
    
    generate_subtitles_if_needed(tmp_wav, req.language, req.want_subs)
    filename = os.path.basename(tmp_wav)
    
    return {
        "status": "success",
        "audio_url": f"/audio/{filename}",
        "subtitles_generated": req.want_subs
    }

# ---------------------------------------------------------------------------
# Gradio Front-End Config & Elements
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", 
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", 
    "[dissatisfaction-hnn]"
]

INSERT_TAG_JS_VC = """(tag_val, current_text) => { const textarea = document.querySelector('#vc_textbox textarea'); if (!textarea) return current_text + " " + tag_val; const start = textarea.selectionStart; const end = textarea.selectionEnd; let prefix = " ", suffix = " "; if (!current_text) return tag_val; if (start === 0) prefix = ""; else if (current_text[start - 1] === ' ') prefix = ""; if (end < current_text.length && current_text[end] === ' ') suffix = ""; return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end); }"""
INSERT_TAG_JS_VD = """(tag_val, current_text) => { const textarea = document.querySelector('#vd_textbox textarea'); if (!textarea) return current_text + " " + tag_val; const start = textarea.selectionStart; const end = textarea.selectionEnd; let prefix = " ", suffix = " "; if (!current_text) return tag_val; if (start === 0) prefix = ""; else if (current_text[start - 1] === ' ') prefix = ""; if (end < current_text.length && current_text[end] === ' ') suffix = ""; return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end); }"""

_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
_CATEGORIES = {"Gender": ["Male", "Female"], "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"], "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"], "Style": ["Whisper"], "English Accent": ["American Accent", "Australian Accent", "British Accent", "Chinese Accent", "Canadian Accent", "Indian Accent", "Korean Accent", "Portuguese Accent", "Russian Accent", "Japanese Accent"], "Chinese Dialect": ["Henan Dialect", "Shaanxi Dialect", "Sichuan Dialect", "Guizhou Dialect", "Yunnan Dialect", "Guilin Dialect", "Jinan Dialect", "Shijiazhuang Dialect", "Gansu Dialect", "Ningxia Dialect", "Qingdao Dialect", "Northeast Dialect"]}
DIALECT_MAP = {"Henan Dialect": "河南话", "Shaanxi Dialect": "陕西话", "Sichuan Dialect": "四川话", "Guizhou Dialect": "贵州话", "Yunnan Dialect": "云南话", "Guilin Dialect": "桂林话", "Jinan Dialect": "济南话", "Shijiazhuang Dialect": "石家庄话", "Gansu Dialect": "甘肃话", "Ningxia Dialect": "宁夏话", "Qingdao Dialect": "青岛话", "Northeast Dialect": "东北话"}
_ATTR_INFO = {"English Accent": "Only effective for English speech.", "Chinese Dialect": "Only effective for Chinese speech."}

theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = ".gradio-container {max-width: 100% !important; font-size: 16px !important;} .gradio-container h1 {font-size: 1.5em !important;} .tag-container {display: flex !important; flex-wrap: wrap !important; gap: 8px !important; margin-top: 5px !important; margin-bottom: 10px !important; border: none !important; background: transparent !important;} .tag-btn {min-width: fit-content !important; width: auto !important; height: 32px !important; font-size: 13px !important; background: #eef2ff !important; border: 1px solid #c7d2fe !important; color: #3730a3 !important; border-radius: 6px !important; padding: 0 10px !important; margin: 0 !important; box-shadow: none !important;} .tag-btn:hover {background: #c7d2fe !important; transform: translateY(-1px);}"

def _lang_dropdown(label="Language (optional)", value="Auto"):
    return gr.Dropdown(label=label, choices=_ALL_LANGUAGES, value=value, allow_custom_value=False, interactive=True)

def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
        du = gr.Number(value=None, label="Duration (seconds)")
        ns = gr.Slider(4, 64, value=32, step=1, label="Inference Steps")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
        pp = gr.Checkbox(label="Preprocess Prompt", value=True)
        po = gr.Checkbox(label="Postprocess Output", value=True)
    return ns, gs, dn, sp, du, pp, po

with gr.Blocks(theme=theme, css=css, title="OmniVoice Server") as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ OmniVoice API Server</h1>
            <p>UI Accessible directly, API Endpoints exposed transparently at <b>/docs</b></p>
        </div>
    """)

    with gr.Tabs():
        # --- Voice Clone Tab ---
        with gr.TabItem("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(label="Text to Synthesize", lines=4, elem_id="vc_textbox")
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(fn=None, inputs=[btn, vc_text], outputs=vc_text, js=INSERT_TAG_JS_VC)
                    with gr.Row():
                      vc_lang = _lang_dropdown()
                      vc_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vc_ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                    vc_ref_text = gr.Textbox(label="Reference Text", lines=2)
                    vc_btn = gr.Button("Generate", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="Output Audio", type="numpy")
                    vc_status = gr.Textbox(label="Status", lines=1)
                    with gr.Accordion("Download files", open=False):
                        vc_out_wav = gr.File(label="WAV")
                        vc_out_custom_srt = gr.File(label="Sentence SRT")
                        vc_out_word_srt = gr.File(label="Word SRT")
                        vc_out_shorts_srt = gr.File(label="Shorts SRT")

            def _auto_transcribe(audio_path, lang):
                if not audio_path or subtitle_maker is None: return gr.update(value="")
                try:
                    whisper_results = subtitle_maker(audio_path, None if lang == "Auto" else lang)
                    if whisper_results and len(whisper_results) > 7: return gr.update(value=whisper_results[7])
                except Exception: pass
                return gr.update(value="")

            vc_ref_audio.change(fn=_auto_transcribe, inputs=[vc_ref_audio, vc_lang], outputs=[vc_ref_text])

            def _clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po):
                res = _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text)
                if res[0] is None: return None, res[1], None, None, None, None
                sr, waveform = res[0]
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return res[0], res[1], tmp_wav, c_srt, w_srt, s_srt

            vc_btn.click(_clone_fn, inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po], outputs=[vc_audio, vc_status, vc_out_wav, vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt])

        # --- Voice Design Tab ---
        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(label="Text to Synthesize", lines=4, elem_id="vd_textbox")
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(fn=None, inputs=[btn, vd_text], outputs=vd_text, js=INSERT_TAG_JS_VD)
                    with gr.Row():
                      vd_lang = _lang_dropdown()
                      vd_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vd_btn = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Character Voice Design", open=False):
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            default_val = "Female" if _cat == "Gender" else "Young Adult" if _cat == "Age" else "Auto"
                            vd_groups.append(gr.Dropdown(label=_cat, choices=["Auto"] + _choices, value=default_val, info=_ATTR_INFO.get(_cat)))
                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="Output Audio", type="numpy")
                    vd_status = gr.Textbox(label="Status", lines=1)
                    with gr.Accordion("Download files", open=False):
                        vd_out_wav = gr.File(label="WAV")
                        vd_out_custom_srt = gr.File(label="Sentence SRT")
                        vd_out_word_srt = gr.File(label="Word SRT")
                        vd_out_shorts_srt = gr.File(label="Shorts SRT")

            def _build_instruct(groups):
                selected = [g for g in groups if g and g != "Auto"]
                return ", ".join([DIALECT_MAP.get(v, v) for v in selected]) if selected else None

            def _design_fn(text, lang, want_subs, ns, gs, dn, sp, du, pp, po, *groups):
                instruct = _build_instruct(groups)
                res = _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")
                if res[0] is None: return None, res[1], None, None, None, None
                sr, waveform = res[0]
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return res[0], res[1], tmp_wav, c_srt, w_srt, s_srt

            vd_btn.click(_design_fn, inputs=[vd_text, vd_lang, vd_want_subs, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups, outputs=[vd_audio, vd_status, vd_out_wav, vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt])

# ---------------------------------------------------------------------------
# App Ingress Mounting & Launch
# ---------------------------------------------------------------------------
app = gr.mount_gradio_app(app, demo, path="/")

def start_localtunnel(port, subdomain):
    """Starts localtunnel in the background with a specific subdomain."""
    logger.info(f"Starting Localtunnel on port {port} with subdomain: {subdomain}...")
    try:
        process = subprocess.Popen(
            ["npx", "localtunnel", "--port", str(port), "--subdomain", subdomain],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            if "your url is" in line.lower():
                url = line.strip().split(" ")[-1]
                print(f"\n=======================================================")
                print(f"🌍 PUBLIC API & UI URL: {url}")
                print(f"⚡ API DOCS: {url}/docs")
                print(f"=======================================================\n")
                break
    except Exception as e:
        logger.error(f"Failed to start localtunnel: {e}")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # IMPORTANT: CHANGE THIS TO YOUR DESIRED UNIQUE NAME!
    # ---------------------------------------------------------
    SUBDOMAIN = "my-unique-omnivoice-api"  # E.g., 'omni-tts-123'
    PORT = 8000
    
    # Start localtunnel in a background thread
    threading.Thread(target=start_localtunnel, args=(PORT, SUBDOMAIN), daemon=True).start()

    # Start FastAPI server
    logger.info(f"Starting API Server on host 0.0.0.0 and port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
