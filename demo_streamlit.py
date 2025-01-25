import streamlit as st
from audio_recorder_streamlit import audio_recorder

# Title for the app
st.title("Audio Recorder and Playback")

# Record audio using streamlit_audio_recorder
audio_data = audio_recorder()
