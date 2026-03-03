import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import os
import time
import base64
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from moviepy import AudioFileClip, VideoClip, ImageClip, concatenate_videoclips
from dotenv import load_dotenv
from youtube_uploader import upload_video
from drive_uploader import upload_to_drive

app = Flask(__name__, static_folder='../', static_url_path='')
CORS(app)

# Load local .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# OpenClaw Integration
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:18789")
GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
TELEGRAM_TARGET = os.environ.get("TELEGRAM_TARGET", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ADAM_VOICE_ID = os.environ.get("ELEVENLABS_ADAM_VOICE_ID", "pNInz6obpg8ndEArWgg8")
BRIAN_VOICE_ID = os.environ.get("ELEVENLABS_BRIAN_VOICE_ID", "nPczCjzI2devNBz1zQrb")
DEFAULT_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", BRIAN_VOICE_ID)

YOUTUBE_UPLOAD_ENABLED = os.environ.get("YOUTUBE_UPLOAD_ENABLED", "true").lower() == "true"
DEBUG_UPLOADS = os.environ.get("DEBUG_UPLOADS", "true").lower() == "true"
YOUTUBE_CLIENT_SECRETS_FILE = os.environ.get("YOUTUBE_CLIENT_SECRETS_FILE", "client_secret.json")
YOUTUBE_TOKEN_FILE = os.environ.get("YOUTUBE_TOKEN_FILE", "youtube_token.json")
YOUTUBE_PRIVACY_STATUS = os.environ.get("YOUTUBE_PRIVACY_STATUS", "private")
YOUTUBE_CATEGORY_ID = os.environ.get("YOUTUBE_CATEGORY_ID", "22")
YOUTUBE_TITLE_PREFIX = os.environ.get("YOUTUBE_TITLE_PREFIX", "Text to Video with Speach")
YOUTUBE_DESCRIPTION = os.environ.get("YOUTUBE_DESCRIPTION", "")

DRIVE_UPLOAD_ENABLED = os.environ.get("DRIVE_UPLOAD_ENABLED", "true").lower() == "true"
DRIVE_CLIENT_SECRETS_FILE = os.environ.get("DRIVE_CLIENT_SECRETS_FILE", "client_secret.json")
DRIVE_TOKEN_FILE = os.environ.get("DRIVE_TOKEN_FILE", "drive_token.pickle")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")

def generate_elevenlabs_audio(text, output_path, voice_id=DEFAULT_VOICE_ID):
    if not ELEVENLABS_API_KEY:
        return False
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        response = requests.post(url, json=data, headers=headers)
        log_path = os.path.join(os.path.dirname(__file__), "elevenlabs_debug.log")
        with open(log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(f"{time.ctime()}: ElevenLabs Response {response.status_code} - {response.text}\n")
        print(f"ElevenLabs API Response: {response.status_code}")
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"ElevenLabs API Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"ElevenLabs Request Failed: {e}")
        return False

def send_to_telegram(file_path, caption):
    if not GATEWAY_TOKEN or not TELEGRAM_TARGET:
        return
    try:
        url = f"{GATEWAY_URL}/api/message"
        headers = {"Authorization": f"Bearer {GATEWAY_TOKEN}"}
        payload = {
            "action": "send",
            "target": TELEGRAM_TARGET,
            "message": caption,
            "filePath": os.path.abspath(file_path)
        }
        requests.post(url, json=payload, headers=headers)
    except Exception as e:
        print(f"Failed to send to Telegram: {e}")

# Supported tld (Top Level Domains) for different accents in gTTS
ACCENTS = {
    "us": "com",      # English (United States)
    "uk": "co.uk",    # English (United Kingdom)
    "au": "com.au",   # English (Australia)
    "ca": "ca",       # English (Canada)
    "in": "co.in",    # English (India)
    "ie": "ie",       # English (Ireland)
    "za": "co.za"     # English (South Africa)
}

def extract_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        return str(e)

def decode_intro_image(intro_image_data: str, timestamp: int):
    if not intro_image_data or not isinstance(intro_image_data, str):
        return None
    try:
        if ',' in intro_image_data:
            _, encoded = intro_image_data.split(',', 1)
        else:
            encoded = intro_image_data
        image_bytes = base64.b64decode(encoded)
        out_path = os.path.join(app.static_folder, f"intro_{timestamp}.png")
        with open(out_path, 'wb') as f:
            f.write(image_bytes)
        return out_path
    except Exception as e:
        print(f"Failed to decode intro image: {e}")
        return None


def choose_font_size_from_word_count(raw_text: str) -> int:
    words = max(1, len((raw_text or '').split()))

    # Linear scale requested:
    # 1 word -> 40px
    # 12000 words -> 10px
    min_words, max_words = 1, 12000
    max_size, min_size = 40.0, 10.0

    clamped_words = max(min_words, min(max_words, words))
    ratio = (clamped_words - min_words) / (max_words - min_words)
    size = max_size + ratio * (min_size - max_size)

    return int(round(size * 0.6))


def make_text_frame(t, total_duration, raw_text, size, font):
    import re

    img = Image.new('RGB', size, color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    margin = 40
    width = size[0] - (margin * 2)
    # Dynamic line height prevents overlap at large font sizes
    if hasattr(font, 'size'):
        line_height = max(18, int(font.size * 1.45))
    else:
        line_height = 24
    top_padding = max(24, int(line_height * 0.9))

    # Sentence-based parsing for more accurate highlighting
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text.replace('\n', ' ')) if s.strip()]
    if not sentences:
        sentences = [raw_text.strip() or ""]

    # Character-weight timing per sentence
    weights = [max(1, len(s)) for s in sentences]
    total_weight = sum(weights)

    sync_factor = 1.0
    highlight_delay = 0.35
    effective_time = max(0.0, (t * sync_factor) - highlight_delay)
    progress_weight = (effective_time / total_duration) * total_weight if total_duration > 0 else 0

    running = 0
    active_sentence_idx = 0
    for i, w in enumerate(weights):
        running += w
        if progress_weight <= running:
            active_sentence_idx = i
            break
    else:
        active_sentence_idx = max(0, len(sentences) - 1)

    # Highlight only the sentence currently being read aloud
    highlight_set = {active_sentence_idx}

    # Wrap sentences into lines while preserving sentence index
    lines = []
    line_sentence_idx = []
    for s_idx, sentence in enumerate(sentences):
        words = sentence.split()
        if not words:
            continue
        current = []
        for word in words:
            test_line = " ".join(current + [word])
            if draw.textlength(test_line, font=font) <= width:
                current.append(word)
            else:
                lines.append(" ".join(current))
                line_sentence_idx.append(s_idx)
                current = [word]
        if current:
            lines.append(" ".join(current))
            line_sentence_idx.append(s_idx)

    if not lines:
        return np.array(img)

    # Scroll to keep active sentence area in view
    active_line_candidates = [i for i, s_idx in enumerate(line_sentence_idx) if s_idx == active_sentence_idx]
    active_line_idx = active_line_candidates[0] if active_line_candidates else 0
    visible_lines = max(1, int((size[1] - top_padding * 2) / line_height))
    start_line = max(0, active_line_idx - max(2, visible_lines // 3))
    end_line = min(len(lines), start_line + visible_lines)

    y = top_padding
    for i in range(start_line, end_line):
        line = lines[i]
        s_idx = line_sentence_idx[i]
        is_highlight = s_idx in highlight_set

        if is_highlight:
            line_w = draw.textlength(line, font=font)
            draw.rectangle([margin - 6, y - 2, margin + line_w + 6, y + line_height - 2], fill=(45, 65, 95))
            color = (235, 245, 255)
        else:
            color = (160, 160, 160)

        draw.text((margin, y), line, font=font, fill=color)
        y += line_height

    return np.array(img)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def generate_openai_audio(text, output_path):
    if not OPENAI_API_KEY:
        return False
    try:
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1-hd",
            "input": text,
            "voice": "onyx"
        }
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            log_path = os.path.join(os.path.dirname(__file__), "openai_debug.log")
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"{time.ctime()}: OpenAI Response {response.status_code} - {response.text}\n")
            return False
    except Exception as e:
        print(f"OpenAI Request Failed: {e}")
        return False

@app.route('/api/generate-video', methods=['POST'])
def api_generate_video():
    data = request.json
    text = data.get('text', '')[:72000]  # allow up to 12k words (~72k chars)
    accent_key = data.get('accent', 'us')
    voice_id = data.get('voiceId', DEFAULT_VOICE_ID)
    youtube_title = (data.get('youtubeTitle') or '').strip()
    youtube_description = (data.get('youtubeDescription') or '').strip()
    upload_to_youtube = bool(data.get('uploadToYoutube', False))
    upload_to_tiktok = bool(data.get('uploadToTiktok', False))
    should_upload_to_drive = bool(data.get('uploadToDrive', True))  # Default to True
    intro_image_data = data.get('introImageData')
    allow_fallback = bool(data.get('allowFallback', False))
    tld = ACCENTS.get(accent_key, 'com')

    resolution = str(data.get('resolution', '1080p')).lower()

    format_maps = {
        "720p": {
            "portrait_tiktok": (720, 1280),
            "landscape_720p": (1280, 720),
            "square_instagram": (720, 720)
        },
        "1080p": {
            "portrait_tiktok": (1080, 1920),
            "landscape_720p": (1920, 1080),
            "square_instagram": (1080, 1080)
        },
        "2k": {
            "portrait_tiktok": (1440, 2560),
            "landscape_720p": (2560, 1440),
            "square_instagram": (1440, 1440)
        },
        "4k": {
            "portrait_tiktok": (2160, 3840),
            "landscape_720p": (3840, 2160),
            "square_instagram": (2160, 2160)
        }
    }

    format_map = format_maps.get(resolution, format_maps["1080p"])
    requested_formats = data.get('formats', ['portrait_tiktok'])
    if not isinstance(requested_formats, list):
        requested_formats = ['portrait_tiktok']
    selected = [(name, *format_map[name]) for name in requested_formats if name in format_map]
    if not selected:
        selected = [('portrait_tiktok', *format_map['portrait_tiktok'])]
    
    if not text: return jsonify({"error": "No text provided"}), 400
    
    timestamp = int(time.time())
    audio_path = f"audio_{timestamp}.mp3"
    intro_image_path = decode_intro_image(intro_image_data, timestamp)

    try:
        # Try ElevenLabs (selected voice) first, then OpenAI, then gTTS
        clean_text_for_tts = text.replace('\n', ' ')
        success = generate_elevenlabs_audio(clean_text_for_tts, audio_path, voice_id)

        if not success:
            if not allow_fallback:
                return jsonify({
                    "error": "ElevenLabs failed. Do you want to continue with fallback voices (OpenAI/gTTS)?",
                    "elevenlabsFailed": True,
                    "canProceedWithoutElevenlabs": True
                }), 409
            print("Falling back to OpenAI TTS...")
            success = generate_openai_audio(clean_text_for_tts, audio_path)

        if not success:
            print("Falling back to gTTS...")
            tts = gTTS(text=clean_text_for_tts, lang='en', tld=tld)
            tts.save(audio_path)
        
        audio = AudioFileClip(audio_path)
        duration = audio.duration
        font_size = choose_font_size_from_word_count(text)
        print(f"Word-count font sizing: words={len(text.split())}, font_size={font_size}")
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        outputs = []
        
        for name, w, h in selected:
            filename = f"{name}_{timestamp}.mp4"
            output_path = os.path.join(app.static_folder, filename)
            # Pass the raw text to preserve formatting
            text_video = VideoClip(lambda t: make_text_frame(t, duration, text, (w, h), font), duration=duration)
            text_video = text_video.with_audio(audio)

            final_video = text_video
            intro_fit_path = None
            if intro_image_path:
                try:
                    # If an image is uploaded, show it for full duration and slowly scroll top -> bottom when tall
                    src_img = Image.open(intro_image_path).convert('RGB')
                    src_w, src_h = src_img.size

                    scale = w / src_w
                    scaled_w = w
                    scaled_h = max(1, int(src_h * scale))
                    scaled_img = src_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

                    if scaled_h <= h:
                        # Fits in frame: letterbox/pad and keep static
                        intro_fit_path = os.path.join(app.static_folder, f"intro_{name}_{timestamp}.png")
                        fitted = ImageOps.pad(scaled_img, (w, h), method=Image.Resampling.LANCZOS, color=(0, 0, 0))
                        fitted.save(intro_fit_path)
                        final_video = ImageClip(intro_fit_path, duration=duration).with_audio(audio)
                    else:
                        # Taller than frame: smooth linear scroll from top to bottom across full audio duration
                        full_arr = np.array(scaled_img)
                        max_offset = scaled_h - h

                        def make_scroll_frame(t):
                            progress = 0 if duration <= 0 else min(1.0, max(0.0, t / duration))
                            y_off = int(progress * max_offset)
                            return full_arr[y_off:y_off + h, 0:w]

                        final_video = VideoClip(make_scroll_frame, duration=duration).with_audio(audio)
                except Exception as intro_err:
                    print(f"Intro image step failed for {filename}: {intro_err}")
                    final_video = text_video

            final_video.write_videofile(output_path, fps=8, codec="libx264", audio_codec="aac", preset="ultrafast")
            
            # Proactively send to Telegram
            send_to_telegram(output_path, f"🎬 Render Complete (V3): {name}")

            outputs.append({"type": name, "url": "/" + filename, "filename": filename})
        
        audio.close()
        if os.path.exists(audio_path): os.remove(audio_path)
        if intro_image_path and os.path.exists(intro_image_path):
            os.remove(intro_image_path)
        for name, _, _ in selected:
            tmp_intro = os.path.join(app.static_folder, f"intro_{name}_{timestamp}.png")
            if os.path.exists(tmp_intro):
                os.remove(tmp_intro)
        return jsonify({"videos": outputs})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-video', methods=['POST'])
def api_upload_video():
    data = request.json or {}
    filename = data.get('filename', '')
    targets = data.get('targets', [])  # list of "youtube", "drive"
    title = (data.get('title') or YOUTUBE_TITLE_PREFIX).strip()
    description = (data.get('description') or YOUTUBE_DESCRIPTION).strip()

    if DEBUG_UPLOADS:
        print(f"[UPLOAD DEBUG] request={data}")

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    file_path = os.path.join(app.static_folder, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"Video file not found: {file_path}"}), 404

    results = {
        "debug": {
            "targets": targets,
            "youtubeUploadEnabled": YOUTUBE_UPLOAD_ENABLED,
            "driveUploadEnabled": DRIVE_UPLOAD_ENABLED,
            "filePath": file_path,
            "fileExists": os.path.exists(file_path),
            "clientSecrets": os.path.join(app.static_folder, YOUTUBE_CLIENT_SECRETS_FILE),
            "tokenFile": os.path.join(app.static_folder, YOUTUBE_TOKEN_FILE),
            "clientSecretsExists": os.path.exists(os.path.join(app.static_folder, YOUTUBE_CLIENT_SECRETS_FILE)),
            "tokenFileExists": os.path.exists(os.path.join(app.static_folder, YOUTUBE_TOKEN_FILE)),
        }
    }

    if "drive" in targets:
        if not DRIVE_UPLOAD_ENABLED:
            results["driveError"] = "Drive upload is disabled on server (DRIVE_UPLOAD_ENABLED=false)"
        else:
            try:
                print(f"Starting Google Drive upload for {filename}...")
                drive_url, drive_id = upload_to_drive(
                    file_path=file_path,
                    title=filename,
                    description=f"{title}\n\n{description}",
                    folder_id=DRIVE_FOLDER_ID or None,
                    client_secrets_file=os.path.join(app.static_folder, DRIVE_CLIENT_SECRETS_FILE),
                    token_file=os.path.join(app.static_folder, DRIVE_TOKEN_FILE)
                )
                results["driveUrl"] = drive_url
                print(f"Google Drive upload complete: {drive_url}")
            except Exception as e:
                print(f"Google Drive upload error: {e}")
                if DEBUG_UPLOADS:
                    print(traceback.format_exc())
                results["driveError"] = str(e)

    if "youtube" in targets:
        if not YOUTUBE_UPLOAD_ENABLED:
            results["youtubeError"] = "YouTube upload is disabled on server (YOUTUBE_UPLOAD_ENABLED=false)"
        else:
            try:
                print(f"Starting YouTube upload for {filename}...")
                youtube_url, _ = upload_video(
                    file_path=file_path,
                    title=title,
                    description=description,
                    tags=["text to video", "tts"],
                    category_id=YOUTUBE_CATEGORY_ID,
                    privacy_status=YOUTUBE_PRIVACY_STATUS,
                    client_secrets_file=os.path.join(app.static_folder, YOUTUBE_CLIENT_SECRETS_FILE),
                    token_file=os.path.join(app.static_folder, YOUTUBE_TOKEN_FILE),
                )
                results["youtubeUrl"] = youtube_url
                print(f"YouTube upload complete: {youtube_url}")
            except Exception as e:
                print(f"YouTube upload error: {e}")
                if DEBUG_UPLOADS:
                    print(traceback.format_exc())
                results["youtubeError"] = str(e)


    if DEBUG_UPLOADS:
        print(f"[UPLOAD DEBUG] response={results}")

    return jsonify(results)

@app.route('/api/version', methods=['GET'])
def api_version():
    return jsonify({"version": "v3", "name": "video-reader-restart-v3", "youtubeUploadEnabled": YOUTUBE_UPLOAD_ENABLED})

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(port=5000)

