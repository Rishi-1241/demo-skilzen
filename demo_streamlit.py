import streamlit as st
import numpy as np
from scipy.io.wavfile import write
import tempfile
from google.cloud import speech, texttospeech
import io
import google.generativeai as genai
from google.cloud import speech
from google.cloud import texttospeech
import os
import numpy as np
from scipy.io import wavfile
import time
import asyncio
import time
from google.cloud import speech, texttospeech
import uuid
from google.cloud.dialogflowcx_v3 import AgentsClient, SessionsClient
from google.cloud.dialogflowcx_v3.types import session
import audio_recorder_streamlit as audio_recorder

GOOGLE_API_KEY = "AIzaSyCnP8cuR-cfpydBlHYbFv0fAtJhqdpHQKQ"
project_id = "certain-math-447716-d1"  # Your Google Cloud project ID
location_id = "global"  # Your agent's location, e.g., "global"
agent_id = "ab039e5f-d9ce-4feb-90ad-4184f23f01e5"  # Your Dialogflow CX agent ID
flow_id = "dd90ab06-761a-410d-bb04-f60368c323ac"
agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"
GOOGLE_APPLICATION_CREDENTIALS = fb_credentials = st.secrets["firebase"]['my_project_settings']
import io
import wave

# Function to get the sample rate from the audio data
def get_sample_rate(audio_data):
    with wave.open(io.BytesIO(audio_data), "rb") as audio_file:
        return audio_file.getframerate()

    # Use a unique session ID for the interaction
session_id = uuid.uuid4()
genai.configure(api_key=GOOGLE_API_KEY)
LANGUAGE_CONFIGS = {
    "English": {
        "code": "en-US",
        "voice_name": "en-US-Wavenet-F",
        "fallback_voice": "en-US-Standard-C"
    },
    "Hindi": {
        "code": "hi-IN",
        "voice_name": "hi-IN-Wavenet-A",
        "fallback_voice": "hi-IN-Standard-A"
    },
    "Telugu": {
        "code": "te-IN",
        "voice_name": "te-IN-Standard-A",
        "fallback_voice": "te-IN-Standard-A"
    },
    "Tamil": {
        "code": "ta-IN",
        "voice_name": "ta-IN-Wavenet-A",
        "fallback_voice": "ta-IN-Standard-A"
    },
    "Kannada": {
        "code": "kn-IN",
        "voice_name": "kn-IN-Wavenet-A",
        "fallback_voice": "kn-IN-Standard-A"
    },
    "Malayalam": {
        "code": "ml-IN",
        "voice_name": "ml-IN-Wavenet-A",
        "fallback_voice": "ml-IN-Standard-A"
    }
}

def get_dialogflow_response(text, language_code, agent, session_id, flow_id):
    """Get response from Dialogflow CX agent"""
    environment_id = "draft"  # Or use "production" if appropriate
    session_path = f"{agent}/environments/{environment_id}/sessions/{session_id}?flow={flow_id}"

    # Prepare text input for Dialogflow
    text_input = session.TextInput(text=text)
    query_input = session.QueryInput(text=text_input, language_code=language_code)

    # Create a detect intent request
    request = session.DetectIntentRequest(
        session=session_path,
        query_input=query_input,
    )

    # Create a session client
    session_client = SessionsClient()

    # Call the API
    response = session_client.detect_intent(request=request)

    # Get the response messages
    response_messages = [
        " ".join(msg.text.text) for msg in response.query_result.response_messages
    ]
    return " ".join(response_messages)
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
from google.cloud import speech
from google.cloud import texttospeech

# Initialize Google Cloud clients
stt_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

def convert_to_mono(audio_data):
    # Read the audio from the WAV data
    sample_rate, audio_array = wavfile.read(io.BytesIO(audio_data))
    
    # Check if the audio is stereo (2 channels)
    if audio_array.ndim == 2:
        # Convert stereo to mono by averaging the two channels
        mono_audio = np.mean(audio_array, axis=1).astype(np.int16)
    else:
        # If already mono, return as is
        mono_audio = audio_array

    # Write the mono audio to a bytes buffer
    with io.BytesIO() as mono_wav:
        wavfile.write(mono_wav, sample_rate, mono_audio)
        mono_wav.seek(0)
        return mono_wav.read()


# Function to transcribe audio using Google STT
def transcribe_audio(audio_data):
    st.info("Transcribing audio...")
    sample_rate = get_sample_rate(audio_data)
    audio_content = audio_data
    mono_audio_data = convert_to_mono(audio_data)
    audio = speech.RecognitionAudio(content=mono_audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
    )
    response = stt_client.recognize(config=config, audio=audio)
    transcription = response.results[0].alternatives[0].transcript if response.results else "Could not transcribe audio"
    st.success(f"Transcription: {transcription}")
    return transcription

# Function to generate TTS audio using Google TTS
def generate_audio_response(text):
    st.info("Generating audio response...")
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response.audio_content

# Main Streamlit App
st.title("Simple Audio Interaction with AI")

# Record audio using audio_recorder
audio_data = audio_recorder()

if audio_data:
    # Display the recorded audio and allow playback
    st.audio(audio_data, format="audio/wav")

    # Transcribe audio
    transcription = transcribe_audio(audio_data)

    # Generate response (You can replace this with your model's output)
    response_text = get_dialogflow_response(transcription,"en-US",agent,session_id,flow_id)  # Replace with your model's processing
    st.success(f"Response: {response_text}")

    # Convert response to audio
    response_audio = generate_audio_response(response_text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_response_audio:
        temp_response_audio.write(response_audio)
        st.audio(temp_response_audio.name, format="audio/wav")
