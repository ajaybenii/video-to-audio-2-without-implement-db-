from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import asyncio
import json
import os
import logging
from websockets.legacy.client import connect
from datetime import datetime, timedelta
import time
from collections import deque
import uuid
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from pydantic import BaseModel

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent

# Configure Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Backend Logger (app.log)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Frontend Logger (frontend.log)
frontend_logger = logging.getLogger("frontend")
frontend_logger.setLevel(logging.INFO)
frontend_handler = logging.FileHandler(LOG_DIR / "frontend.log")
frontend_handler.setFormatter(logging.Formatter("%(asctime)s [FRONTEND] %(message)s"))
frontend_logger.addHandler(frontend_handler)

# Import for service account authentication
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Configuration (from environment variables)
PROJECT_ID = os.getenv("PROJECT_ID", "sqy-prod")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_ID = "gemini-live-2.5-flash-native-audio"
MODEL = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}"
HOST = f"wss://{LOCATION}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

# 🔥 PRODUCTION SETTINGS
MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MAX_CONCURRENT_CONNECTIONS", "1000"))
CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", "1800"))  # 30 minutes

INTERVIEW_PROMPT = """
You are conducting a real-time technical interview for a Software Engineer position.
You are based in INDIA and conducting this interview in Indian Standard Time (IST, UTC+5:30).

You can hear and also see the candidate through audio and video.

# 🔴 PROCTORING & MONITORING (HIGH PRIORITY):
You MUST continuously monitor the video feed and IMMEDIATELY warn the candidate if you detect:

1. **Multiple People Detected**: If you see more than ONE person in the frame:
   - Immediately say: "I notice there might be someone else in the room. For interview integrity, please ensure you are alone. This will be noted."

2. **Mobile Phone Usage**: If you see the candidate using or looking at a mobile phone:
   - Immediately say: "I noticed you looking at your phone. Please keep your phone away during the interview. Using external devices is not allowed."

3. **Candidate Not Visible**: If the candidate is not visible or has moved out of frame:
   - Immediately say: "I can't see you on the screen. Please adjust your camera so I can see you clearly."

4. **Looking Away / Not Focused**: If the candidate is frequently looking away from the screen (looking left, right, up, or down repeatedly):
   - Say: "I notice you're looking away from the screen. Please focus on the interview and maintain eye contact with the camera."

5. **Suspicious Behavior**: If you see any suspicious behavior like reading from another screen, someone whispering, or unusual movements:
   - Say: "I noticed some unusual activity. Please remember this is a proctored interview and any unfair means will be recorded."

6. **Tab Switching / Distraction**: If the candidate appears distracted or seems to be reading something off-screen:
   - Say: "It seems like you might be looking at something else. Please give your full attention to the interview."

# 🌐 NETWORK MONITORING:
If you receive a message indicating the candidate's network quality is POOR:
- Say: "I'm noticing some connectivity issues on your end. If possible, please move to a location with better internet connection for a smoother interview experience."


- If you receive "[SYSTEM] Interview time limit (30 minutes) reached. Ending interview.":
  Say: "We have reached the 30-minute time limit for this interview. Thank you for your time today. The interview is now complete."

# 🔄 IMPORTANT - CONTINUING INTERVIEW:
If the candidate speaks or responds after any warning (including the final warning), you MUST:
- IMMEDIATELY continue the interview as normal
- Do NOT say "the interview has ended" or "we've reached the time limit" (unless the 30-minute timer actually reached zero)
- Do NOT refuse to continue - just pick up where you left off
- Simply acknowledge their response and continue with the next question
The silence warnings are just prompts - if the user responds, the interview continues!

# 🚫 CRITICAL - SILENCE HANDLING:
- If the candidate is silent, WAIT for them to speak. Do not say anything.
- NEVER say "I am waiting for your response".
- NEVER output "[SYSTEM]" tags or internal instruction text.
- If the candidate gives a short response like "okay" or "yes", acknowledge it and continue.
- If you get some unrelated input except english and hindi then reconfirm the question in english or hindi.
- If you are unsure if they finished, ask a relevant follow-up question instead of a generic waiting prompt.

# ⚠️ IMPORTANT: Issue warnings in a FIRM but PROFESSIONAL tone. Do not be rude, but be clear that violations are being noted.

# Interview Structure:
1. Greet the candidate appropriately based on the CURRENT TIME provided in [CURRENT CONTEXT] below:
   - 6 AM - 12 PM: "Good morning"
   - 12 PM - 5 PM: "Good afternoon"  
   - 5 PM - 9 PM: "Good evening"
   - 9 PM - 6 AM: "Hello"

2. Ask candidate to introduce themselves
3. Ask 3 technical questions about:
   - Data structures and algorithms
   - System design
   - Problem-solving approach
4. Ask 2 behavioral questions
5. Close the interview professionally

# Visual Observation Rules:
- You can see the candidate through video
- Answer visual questions ONLY based on what is clearly visible
- If something is not clearly visible, say you are not certain
- Do not guess or assume

# Communication Rules:
- Be professional but friendly (Indian professional context)
- Listen carefully and ask follow-up questions
- Keep responses concise
- Encourage the candidate when they do well
- Use natural, conversational language
- Speak clearly in English (Indian candidates may have regional accents - be patient)
- If user want to switch language then proceed with that language
"""

# Service Account Authentication
def get_access_token():
    """Get access token using service account credentials"""
    try:
        # Load credentials from environment or file
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        else:
            # Load from JSON string in environment variable
            credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
            if credentials_json:
                credentials_info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            else:
                logger.error("No credentials found in environment")
                return None
        
        # Refresh token
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        logger.error(f"Error getting access token: {e}")
        return None

# Lifespan context manager (replaces deprecated @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 60)
    logger.info("VOICE + VIDEO INTERVIEW BOT API")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Max Connections: {MAX_CONCURRENT_CONNECTIONS}")
    logger.info(f"Connection Timeout: {CONNECTION_TIMEOUT}s")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Video Support: ENABLED")
    logger.info("=" * 60)
    logger.info("Server Ready!")
    yield
    # Shutdown
    logger.info("Server shutting down...")

# Initialize FastAPI
app = FastAPI(
    title="Voice + Video Interview Bot API - Production",
    description="High-performance interview bot with unlimited rate limits",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Connection Management
class ConnectionManager:
    def __init__(self):
        self.active_connections = 0
        self.total_connections = 0
        self.connection_history = deque(maxlen=100)
        self.token_cache = None
        self.token_expiry = None
        self.start_time = datetime.now()
        self._token_lock = asyncio.Lock()
    
    def can_accept_connection(self) -> bool:
        return self.active_connections < MAX_CONCURRENT_CONNECTIONS
    
    def add_connection(self):
        self.active_connections += 1
        self.total_connections += 1
        self.connection_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'connected',
            'active': self.active_connections,
            'total': self.total_connections
        })
    
    def remove_connection(self):
        self.active_connections = max(0, self.active_connections - 1)
        self.connection_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'disconnected',
            'active': self.active_connections
        })
    

    async def get_cached_token(self):
        """Cache token to optimize performance"""
        async with self._token_lock:
            now = datetime.now()
            if self.token_cache and self.token_expiry and now < self.token_expiry:
                return self.token_cache
            token = get_access_token()
            if token:
                self.token_cache = token
                self.token_expiry = now + timedelta(minutes=50)
            return token
    
    def get_stats(self):
        uptime = datetime.now() - self.start_time
        return {
            'active_connections': self.active_connections,
            'total_connections': self.total_connections,
            'max_capacity': MAX_CONCURRENT_CONNECTIONS,
            'available_slots': MAX_CONCURRENT_CONNECTIONS - self.active_connections,
            'uptime_seconds': int(uptime.total_seconds()),
            'uptime_formatted': str(uptime).split('.')[0]
        }

manager = ConnectionManager()

async def relay_messages(ws_client: WebSocket, ws_google):
    """Handle bidirectional message relay between client and Gemini"""
    
    # Store session resumption handle
    session_handle = None
    
    async def client2server(source: WebSocket, target):
        """Browser → Gemini (audio + video)"""
        msg_count = 0
        audio_chunk_count = 0
        try:
            while True:
                message = await source.receive_text()
                msg_count += 1
                data = json.loads(message)
                
                # Logging (only in debug mode)
                if 'realtimeInput' in data:
                    audio_chunk_count += 1
                    if audio_chunk_count % 100 == 0:
                        logger.debug(f"Media chunks sent: {audio_chunk_count}")
                else:
                    logger.debug(f"Browser→Gemini message #{msg_count}")
                
                await target.send(message)
        except WebSocketDisconnect:
            logger.debug("Client disconnected from relay")
        except Exception as e:
            logger.error(f"Error client2server: {e}")
    
    async def server2client(source, target: WebSocket):
        """Gemini → Browser"""
        nonlocal session_handle
        msg_count = 0
        try:
            async for message in source:
                msg_count += 1
                data = json.loads(message.decode('utf-8'))
                
                # Handle session resumption updates
                if 'sessionResumptionUpdate' in data:
                    update = data['sessionResumptionUpdate']
                    if update.get('resumable') and update.get('newHandle'):
                        session_handle = update['newHandle']
                        logger.debug("Session resumption handle updated")
                
                # Handle GoAway message (connection will terminate soon)
                if 'goAway' in data:
                    time_left = data['goAway'].get('timeLeft', 'unknown')
                    logger.warning(f"Connection will close in {time_left}. Resumption handle available: {bool(session_handle)}")
                
                # Detailed logging in debug mode
                if 'serverContent' in data:
                    content = data['serverContent']
                    
                    if 'modelTurn' in content:
                        logger.debug("AI Speaking")
                    
                    if 'outputTranscription' in content:
                        text = content['outputTranscription'].get('text', '')
                        logger.debug(f"AI said: {text}")
                    
                    if 'inputTranscription' in content:
                        text = content['inputTranscription'].get('text', '')
                        is_final = content['inputTranscription'].get('isFinal', False)
                        if is_final:
                            logger.debug(f"User said: {text}")
                    
                    if 'generationComplete' in content:
                        logger.debug("Generation complete")
                
                elif 'setupComplete' in data:
                    logger.debug("Setup complete")
                
                await target.send_text(message.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error server2client: {e}")
    
    # Set timeout for the entire connection
    try:
        await asyncio.wait_for(
            asyncio.gather(
                client2server(ws_client, ws_google),
                server2client(ws_google, ws_client),
                return_exceptions=True
            ),
            timeout=CONNECTION_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.warning(f"Connection timeout after {CONNECTION_TIMEOUT} seconds")

@app.get("/")
async def root():
    """API information endpoint"""
    stats = manager.get_stats()
    return {
        "status": "online",
        "service": "Voice + Video Interview Bot API",
        "version": "2.0.0",
        "model": MODEL_ID,
        "features": ["audio", "video", "transcription"],
        "websocket_endpoint": "/ws/interview",
        **stats
    }

@app.get("/health")
async def health_check():
    """Health check for monitoring and load balancers"""
    stats = manager.get_stats()
    is_healthy = stats['active_connections'] < MAX_CONCURRENT_CONNECTIONS
    
    return {
        "status": "healthy" if is_healthy else "at_capacity",
        "video_support": True,
        "rate_limits": "unlimited",
        **stats
    }

@app.get("/stats")
async def get_stats():
    """Detailed statistics endpoint"""
    stats = manager.get_stats()
    return {
        **stats,
        "recent_activity": list(manager.connection_history)[-20:],
        "configuration": {
            "max_concurrent_connections": MAX_CONCURRENT_CONNECTIONS,
            "connection_timeout": CONNECTION_TIMEOUT,
            "model": MODEL_ID,
            "location": LOCATION
        }
    }

@app.websocket("/ws/interview")
async def websocket_interview(websocket: WebSocket):
    """Main WebSocket endpoint for voice + video interview"""
    
    # Check capacity
    if not manager.can_accept_connection():
        await websocket.close(code=1008, reason="Server at capacity")
        logger.warning(f"Connection rejected - At capacity ({manager.active_connections}/{MAX_CONCURRENT_CONNECTIONS})")
        return
    
    await websocket.accept()
    manager.add_connection()
    
    connection_id = manager.total_connections
    
    logger.info(f"Client #{connection_id} connected ({manager.active_connections}/{MAX_CONCURRENT_CONNECTIONS} active)")
    
    # Get cached token for better performance
    access_token = await manager.get_cached_token()
    
    if not access_token:
        logger.error("Failed to get access token")
        manager.remove_connection()
        await websocket.close(code=1011, reason="Authentication failed")
        return
    
    try:
        async with connect(
            HOST,
            extra_headers={'Authorization': f'Bearer {access_token}'},
            ping_interval=20,
            ping_timeout=10,
            max_size=10_000_000  # 10MB max message size for video
        ) as ws_google:
            # Determine current time in India (IST)
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            current_time_str = ist_now.strftime("%A, %d %B %Y, %I:%M %p")
            
            # Dynamic System Prompt with Time Context
            dynamic_prompt = INTERVIEW_PROMPT + f"\n\n[CURRENT CONTEXT]\nCurrent Time (IST): {current_time_str}\nDefault Location: India"

            # Setup with audio and video support + UNLIMITED SESSION TIME
            initial_request = {
                "setup": {
                    "model": MODEL,
                    "generationConfig": {
                        "temperature": 0.7,
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": "Aoede"
                                }
                            }
                        }
                    },
                    "systemInstruction": {
                        "parts": [{"text": dynamic_prompt}]
                    },
                    "input_audio_transcription": {},
                    "output_audio_transcription": {},
                    # 🔥 CRITICAL: Enable context window compression for unlimited session time
                    "context_window_compression": {
                        "sliding_window": {},
                        "trigger_tokens": 50000  # Compress when context reaches 50K tokens
                    },
                    # 🔥 Enable session resumption for handling connection resets
                    "session_resumption": {}
                }
            }
            
            await ws_google.send(json.dumps(initial_request))
            
            logger.debug(f"Client #{connection_id} - AI initialized with video and transcription")
            
            await relay_messages(websocket, ws_google)
            
    except WebSocketDisconnect:
        logger.info(f"Client #{connection_id} disconnected")
    except Exception as e:
        logger.error(f"Client #{connection_id} error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
    finally:
        manager.remove_connection()
        logger.info(f"Client #{connection_id} session ended. Active: {manager.active_connections}/{MAX_CONCURRENT_CONNECTIONS}")



# Network Info Endpoint for latency measurement
@app.get("/api/network-info")
async def network_info():
    """
    Returns server timestamp for client-side latency calculation.
    This endpoint is used by the frontend to measure network quality.
    """
    return {
        "timestamp": int(time.time() * 1000),  # milliseconds
        "status": "ok",
        "server_time": datetime.now().isoformat()
    }


# ============================================================
# SPEED TEST API (Business Reusable)
# ============================================================

# Store speed test results for analytics
speed_test_results = []

@app.get("/api/speed-test/download")
async def speed_test_download(bytes: int = 100000):
    """
    Serve binary data for speed test.
    Client downloads this and measures time to calculate bandwidth.
    Args:
        bytes: Size of test data (default 100KB, max 1MB)
    """
    # Limit max size to 1MB to prevent abuse
    size = min(bytes, 1000000)
    # Generate random bytes for download
    data = os.urandom(size)
    
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(size),
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Speed-Test": "true"
        }
    )

@app.post("/api/speed-test/report")
async def speed_test_report(
    speed_mbps: float,
    quality: str = "unknown",
    user_agent: str = None
):
    """
    Report speed test result for analytics.
    Args:
        speed_mbps: Measured download speed in Mbps
        quality: Network quality (good/fair/poor)
        user_agent: Client user agent
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "speed_mbps": speed_mbps,
        "quality": quality,
        "user_agent": user_agent
    }
    speed_test_results.append(result)
    
    # Keep only last 1000 results in memory
    if len(speed_test_results) > 1000:
        speed_test_results.pop(0)
    
    logger.info(f"Speed test reported: {speed_mbps:.1f} Mbps ({quality})")
    return {"status": "recorded", "speed_mbps": speed_mbps}

@app.get("/api/speed-test/stats")
async def speed_test_stats():
    """
    Get speed test analytics (last 24 hours summary).
    """
    if not speed_test_results:
        return {"count": 0, "avg_speed": 0, "min_speed": 0, "max_speed": 0}
    
    speeds = [r["speed_mbps"] for r in speed_test_results]
    return {
        "count": len(speeds),
        "avg_speed": round(sum(speeds) / len(speeds), 2),
        "min_speed": round(min(speeds), 2),
        "max_speed": round(max(speeds), 2),
        "quality_distribution": {
            "good": sum(1 for r in speed_test_results if r["quality"] == "good"),
            "fair": sum(1 for r in speed_test_results if r["quality"] == "fair"),
            "poor": sum(1 for r in speed_test_results if r["quality"] == "poor")
        }
    }

# ============================================================
# RECORDING & ANALYSIS ENDPOINTS
# ============================================================

# Directory paths for recordings
RECORDINGS_DIR = Path(__file__).parent / "recordings"
AUDIO_DIR = RECORDINGS_DIR / "audio"
AUDIO_USER_DIR = AUDIO_DIR / "user"
AUDIO_COMBINED_DIR = AUDIO_DIR / "combined"
SCREEN_DIR = RECORDINGS_DIR / "screen"
TRANSCRIPTS_DIR = RECORDINGS_DIR / "transcripts"

# Ensure directories exist
for dir_path in [AUDIO_DIR, AUDIO_USER_DIR, AUDIO_COMBINED_DIR, SCREEN_DIR, TRANSCRIPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

import boto3
from botocore.exceptions import NoCredentialsError



# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "edy-temp-videos")

# Initialize S3 Client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)


@app.post("/api/upload-recording")
async def upload_recording(
    file: UploadFile = File(...),
    recording_type: str = "audio"  # audio, combined_audio, or screen
):
    """
    Upload audio or screen recording directly to S3.
    Returns session_id and S3 key.
    """
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine S3 prefix/folder
    # User requested: s3://edy-temp-videos/ai_video_to_audio_nterview/
    base_folder = "ai_video_to_audio_nterview"
    
    if recording_type == "screen":
        subfolder = "screen"
        extension = ".webm"
    elif recording_type == "combined_audio":
        subfolder = "combined_audio"
        extension = ".webm"
    else: # Default "audio" -> User Mic Only
        subfolder = "user_audio"
        extension = ".webm"
    
    filename = f"{session_id}_{timestamp}{extension}"
    s3_key = f"{base_folder}/{subfolder}/{filename}"
    
    try:
        # Upload directly to S3 without saving locally
        # file.file is a SpooledTemporaryFile (file-like object)
        s3_client.upload_fileobj(
            file.file,
            AWS_S3_BUCKET,
            s3_key,
            ExtraArgs={'ContentType': file.content_type}
        )
        
        logger.info(f"Uploaded to S3: s3://{AWS_S3_BUCKET}/{s3_key}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "filename": filename,
            "s3_bucket": AWS_S3_BUCKET,
            "s3_key": s3_key,
            "s3_uri": f"s3://{AWS_S3_BUCKET}/{s3_key}",
            "recording_type": recording_type
        }
        
    except Exception as e:
        logger.error(f"S3 Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"S3 Upload failed: {str(e)}")


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using Vertex AI Gemini 2.5 Flash.
    Returns text file with AI/User separated transcription with timestamps.
    """
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save uploaded audio temporarily
    temp_audio_path = AUDIO_DIR / f"temp_{session_id}.webm"
    content = await file.read()
    with open(temp_audio_path, "wb") as f:
        f.write(content)
    
    try:
        # Use Vertex AI Gemini 2.5 Flash for transcription
        model = GenerativeModel('gemini-2.5-flash')
        
        # Read audio file and create Part
        with open(temp_audio_path, "rb") as f:
            audio_data = f.read()
        audio_part = Part.from_data(audio_data, mime_type="audio/webm")
        
        # Request transcription with speaker separation
        prompt = """
        Transcribe this audio interview conversation.
        
        IMPORTANT: Separate the speakers as follows:
        - AI Interviewer: The AI voice asking questions
        - User: The human candidate answering questions
        
        Format each line as:
        [TIMESTAMP] SPEAKER: Text
        
        Example:
        [00:00:05] AI: Good afternoon, could you please introduce yourself?
        [00:00:12] User: Yes, my name is John and I have 5 years of experience.
        
        Provide accurate timestamps in MM:SS format.
        Transcribe the COMPLETE conversation.
        """
        
        response = model.generate_content([audio_part, prompt])
        transcript_text = response.text
        
        # Save transcript to file
        transcript_filename = f"transcript_{session_id}_{timestamp}.txt"
        transcript_path = TRANSCRIPTS_DIR / transcript_filename
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(f"# Interview Transcript\n")
            f.write(f"# Session ID: {session_id}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            f.write(transcript_text)
        
        # Extract user-only transcript for scoring
        user_lines = []
        for line in transcript_text.split('\n'):
            if 'User:' in line or 'USER:' in line or 'user:' in line:
                user_lines.append(line)
        
        user_transcript = '\n'.join(user_lines)
        
        # Cleanup temp file
        temp_audio_path.unlink(missing_ok=True)
        
        logger.info(f"Transcription complete: {transcript_filename}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "transcript_file": transcript_filename,
            "transcript_path": str(transcript_path),
            "full_transcript": transcript_text,
            "user_transcript": user_transcript
        }
        
    except Exception as e:
        temp_audio_path.unlink(missing_ok=True)
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/score-communication")
async def score_communication(file: UploadFile = File(...)):
    """
    Analyze user's audio for communication skills using Vertex AI.
    Scores: Pitch, Calmness, Fluency, Confidence, Clarity (0-10 total)
    """
    session_id = str(uuid.uuid4())
    
    # Save uploaded audio temporarily
    temp_audio_path = AUDIO_DIR / f"temp_comm_{session_id}.webm"
    content = await file.read()
    with open(temp_audio_path, "wb") as f:
        f.write(content)
    
    try:
        model = GenerativeModel('gemini-2.5-flash')
        
        # Read audio and create Part
        with open(temp_audio_path, "rb") as f:
            audio_data = f.read()
        audio_part = Part.from_data(audio_data, mime_type="audio/webm")
        
        prompt = """
        Analyze this interview audio for the USER/CANDIDATE's communication skills ONLY.
        Ignore the AI interviewer's voice - focus only on the human candidate.
        
        Score each category from 0-2 points:
        
        1. PITCH (0-2): Is the voice pitch appropriate, not too monotone or too varied?
        2. CALMNESS (0-2): How calm and composed does the candidate sound?
        3. FLUENCY (0-2): How smooth is the speech flow? Minimal filler words (um, uh)?
        4. CONFIDENCE (0-2): Does the candidate sound confident and assured?
        5. CLARITY (0-2): How clear and understandable is the speech?
        
        Respond in this EXACT JSON format:
        {
            "pitch": {"score": X, "feedback": "..."},
            "calmness": {"score": X, "feedback": "..."},
            "fluency": {"score": X, "feedback": "..."},
            "confidence": {"score": X, "feedback": "..."},
            "clarity": {"score": X, "feedback": "..."},
            "total_score": X,
            "overall_feedback": "..."
        }
        
        Be strict but fair in scoring.
        """
        
        response = model.generate_content([audio_part, prompt])
        
        # Parse JSON response
        response_text = response.text
        # Clean up response if wrapped in markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        score_data = json.loads(response_text.strip())
        
        # Cleanup
        temp_audio_path.unlink(missing_ok=True)
        
        logger.info(f"Communication score: {score_data.get('total_score', 'N/A')}/10")
        
        return {
            "status": "success",
            "session_id": session_id,
            "score_type": "communication",
            "scores": score_data
        }
        
    except json.JSONDecodeError as e:
        temp_audio_path.unlink(missing_ok=True)
        logger.error(f"JSON parse error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse score response")
    except Exception as e:
        temp_audio_path.unlink(missing_ok=True)
        logger.error(f"Communication scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/api/score-technical")
async def score_technical(file: UploadFile = File(...)):
    """
    Analyze transcript text for technical skills using Vertex AI.
    Scores: Technical accuracy, Problem-solving, Relevance (0-10 total)
    """
    session_id = str(uuid.uuid4())
    
    # Read transcript text file
    content = await file.read()
    transcript_text = content.decode("utf-8")
    
    try:
        model = GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Analyze this interview transcript for the CANDIDATE's technical skills.
        Focus ONLY on the User/Candidate responses, not the AI questions.
        
        TRANSCRIPT:
        {transcript_text}
        
        Score each category:
        
        1. TECHNICAL ACCURACY (0-4): Are the technical answers correct and precise?
        2. PROBLEM SOLVING (0-3): Does the candidate show good problem-solving approach?
        3. RELEVANCE (0-3): Are answers relevant to the questions asked?
        
        Respond in this EXACT JSON format:
        {{
            "technical_accuracy": {{"score": X, "feedback": "..."}},
            "problem_solving": {{"score": X, "feedback": "..."}},
            "relevance": {{"score": X, "feedback": "..."}},
            "total_score": X,
            "overall_feedback": "...",
            "strengths": ["...", "..."],
            "areas_to_improve": ["...", "..."]
        }}
        
        Be strict but fair. Score based on actual technical content.
        """
        
        response = model.generate_content(prompt)
        
        # Parse JSON response
        response_text = response.text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        score_data = json.loads(response_text.strip())
        
        logger.info(f"Technical score: {score_data.get('total_score', 'N/A')}/10")
        
        return {
            "status": "success",
            "session_id": session_id,
            "score_type": "technical",
            "scores": score_data
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse score response")
    except Exception as e:
        logger.error(f"Technical scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.get("/api/recordings/{session_id}")
async def get_recording(session_id: str, recording_type: str = "audio"):
    """
    Get recording file by session ID.
    """
    if recording_type == "screen":
        search_dir = SCREEN_DIR
    else:
        search_dir = AUDIO_DIR
    
    # Find file matching session_id
    for file_path in search_dir.iterdir():
        if session_id in file_path.name:
            return FileResponse(
                path=file_path,
                filename=file_path.name,
                media_type="audio/webm" if recording_type == "audio" else "video/webm"
            )
    
    raise HTTPException(status_code=404, detail="Recording not found")


@app.get("/api/transcript/{session_id}")
async def get_transcript(session_id: str):
    """
    Get transcript file by session ID.
    """
    for file_path in TRANSCRIPTS_DIR.iterdir():
        if session_id in file_path.name:
            return FileResponse(
                path=file_path,
                filename=file_path.name,
                media_type="text/plain"
            )
    
    raise HTTPException(status_code=404, detail="Transcript not found")


class LogEntry(BaseModel):
    level: str
    message: str
    timestamp: str
    context: dict = {}

@app.post("/api/log")
async def receive_frontend_log(entry: LogEntry):
    """
    Receive logs from frontend and save to frontend.log
    """
    msg = f"[{entry.level.upper()}] {entry.message} | Context: {entry.context}"
    if entry.level.lower() == "error":
        frontend_logger.error(msg)
    elif entry.level.lower() == "warn":
        frontend_logger.warning(msg)
    else:
        frontend_logger.info(msg)
    return {"status": "logged"}



if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        limit_concurrency=MAX_CONCURRENT_CONNECTIONS + 50,  # Buffer for safety
        timeout_keep_alive=75,
        ws_ping_interval=20,
        ws_ping_timeout=10
    )
    