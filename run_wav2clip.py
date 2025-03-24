import torch
import torchaudio
from wav2clip import get_model, embed_audio
from transformers import pipeline

# Model wav2clip
model = get_model(device="cpu")
filename = "audio_files/Pitbull_-no-prompt.wav"
waveform, sample_rate = torchaudio.load(filename)

# Convert the sound to embedding
embedding = embed_audio(waveform.numpy(), model)

print("Embedding shape:", embedding.shape)
print("Embedding values:", embedding)

# Generate a caption of the sound

"""
prompt = f"Describe the sound corresponding to this audio embedding: {embedding.tolist()}"
"""
prompt = "The following is a detailed description of the audio:\n"
prompt += f"Describe the sound corresponding to this audio embedding: {embedding.tolist()}\n"
prompt += "The sound is: "

captioner = pipeline("text-generation", model="gpt2", device=-1)

captioner = pipeline("text-generation", model="distilgpt2", device=-1)
response = captioner(prompt)

print(response[0]["generated_text"])

# Truncate the prompt to the maximum length supported by the model
max_length = 1024 - 50
truncated_prompt = prompt[:max_length]

response = captioner(truncated_prompt, max_new_tokens=50, truncation=True)
print(response[0]["generated_text"])

