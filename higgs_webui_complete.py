import gradio as gr
import torch
import torchaudio
import os
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from typing import Optional, List, Tuple
import yaml
import argparse
import sys
import copy
import jieba
import langid
import tqdm
import re
import numpy as np

try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample
except ImportError:
    print("Warning: boson_multimodal not found. Please install the HiggsAudio package.")
    exit(1)

# ===== CONFIGURATION =====
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
OUTPUT_DIR = "generated_audio"
VOICE_PROMPTS_DIR = "voice_prompts"

# Global debug flag
DEBUG_MODE = False

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

# Scene prompts
SCENE_PROMPTS = {
    "Default (Quiet Room)": "Audio is recorded in a quiet room using a Shure SM7B microphone.",
    "Empty": ""
}

# Quality control parameters
QUALITY_CHECK_PARAMS = {
    "MAX_REGENERATION_ATTEMPTS": 3,
    "DURATION_TOLERANCE_FACTOR": 3.0,
    "MIN_DURATION_FACTOR": 0.2,
    "CHARS_PER_SECOND": 15,
    "MIN_AUDIO_LEVEL": 0.005,
    "MIN_RMS_LEVEL": 0.003,
    "CORRELATION_THRESHOLD": 0.8
}

# ===== OFFICIAL HIGGS AUDIO CHUNKING FUNCTIONS =====
def normalize_chinese_punctuation(text):
    """Convert Chinese (full-width) punctuation marks to English (half-width) equivalents."""
    chinese_to_english_punct = {
        "，": ", ",  "。": ".",  "：": ":",  "；": ";",  "？": "?",  "！": "!",
        "（": "(",  "）": ")",  "【": "[",  "】": "]",  "《": "<",  "》": ">",
        """: '"',  """: '"',  "'": "'",  "'": "'",  "、": ",",  "—": "-",
        "…": "...",  "·": ".",  "「": '"',  "」": '"',  "『": '"',  "』": '"',
    }
    
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    
    return text
    
def prepare_chunk_text(
    text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces. Official HiggsAudio implementation."""
    if chunk_method is None or chunk_method == "none":
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            if chunks:
                chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")

def _build_system_message_with_audio_prompt(system_message):
    """Build system message with audio placeholders."""
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    
    return Message(role="system", content=contents)

def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
    """Prepare the context for generation - Official HiggsAudio implementation."""
    system_message = None
    messages = []
    audio_ids = []
    
    if ref_audio is not None:
        num_speakers = len(ref_audio.split(","))
        speaker_info_l = ref_audio.split(",")
        voice_profile = None
        
        if any([speaker_info.startswith("profile:") for speaker_info in ref_audio.split(",")]):
            ref_audio_in_system_message = True
            
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("profile:"):
                    if voice_profile is None:
                        profile_path = os.path.join(VOICE_PROMPTS_DIR, "profile.yaml")
                        if os.path.exists(profile_path):
                            with open(profile_path, "r", encoding="utf-8") as f:
                                voice_profile = yaml.safe_load(f)
                    if voice_profile and "profiles" in voice_profile:
                        character_desc = voice_profile["profiles"].get(character_name[len("profile:"):].strip(), character_name)
                        speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                    else:
                        speaker_desc.append(f"SPEAKER{spk_id}: {character_name}")
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            
            if scene_prompt:
                system_message = (
                    "Generate audio following instruction."
                    "\n\n"
                    f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
                )
            else:
                system_message = (
                    "Generate audio following instruction.\n\n"
                    + f"<|scene_desc_start|>\n"
                    + "\n".join(speaker_desc)
                    + "\n<|scene_desc_end|>"
                )
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )
        
        for spk_id, character_name in enumerate(ref_audio.split(",")):
            if not character_name.startswith("profile:"):
                prompt_audio_path = os.path.join(VOICE_PROMPTS_DIR, f"{character_name}.wav")
                prompt_text_path = os.path.join(VOICE_PROMPTS_DIR, f"{character_name}.txt")
                
                if os.path.exists(prompt_audio_path) and os.path.exists(prompt_text_path):
                    with open(prompt_text_path, "r", encoding="utf-8") as f:
                        prompt_text = f.read().strip()
                    
                    if audio_tokenizer is not None:
                        audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                        audio_ids.append(audio_tokens)

                    if not ref_audio_in_system_message:
                        messages.append(
                            Message(
                                role="user",
                                content=f"[SPEAKER{spk_id}] {prompt_text}" if num_speakers > 1 else prompt_text,
                            )
                        )
                        messages.append(
                            Message(
                                role="assistant",
                                content=AudioContent(audio_url=prompt_audio_path),
                            )
                        )
    else:
        if len(speaker_tags) > 1:
            speaker_desc_l = []
            for idx, tag in enumerate(speaker_tags):
                if idx % 2 == 0:
                    speaker_desc = f"feminine"
                else:
                    speaker_desc = f"masculine"
                speaker_desc_l.append(f"{tag}: {speaker_desc}")

            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = []
            if scene_prompt:
                scene_desc_l.append(scene_prompt)
            scene_desc_l.append(speaker_desc)
            scene_desc = "\n\n".join(scene_desc_l)

            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(
                role="system",
                content="\n\n".join(system_message_l),
            )
    
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids

# ===== QUALITY CONTROL FUNCTIONS =====

def estimate_expected_duration(text_content):
    """Estimate expected audio duration based on text length"""
    clean_text = re.sub(r'\[SPEAKER\d+\]', '', text_content)
    clean_text = re.sub(r'<SE[^>]*>[^<]*</SE[^>]*>', '', clean_text)
    clean_text = re.sub(r'<SE[^>]*>', '', clean_text)
    
    char_count = len(clean_text.strip())
    return max(1.0, char_count / QUALITY_CHECK_PARAMS["CHARS_PER_SECOND"])

def check_audio_quality(audio_array, sampling_rate, expected_duration, text_content, enable_quality_checks=True):
    """Check if generated audio meets quality criteria"""
    if not enable_quality_checks:
        return True, "Quality checks disabled"
    
    if audio_array is None or len(audio_array) == 0:
        return False, "No audio generated"
    
    duration = len(audio_array) / sampling_rate
    audio_max = np.max(np.abs(audio_array))
    audio_rms = np.sqrt(np.mean(audio_array ** 2))
    
    if audio_max < QUALITY_CHECK_PARAMS["MIN_AUDIO_LEVEL"] or audio_rms < QUALITY_CHECK_PARAMS["MIN_RMS_LEVEL"]:
        return False, f"Audio too quiet (max: {audio_max:.4f}, rms: {audio_rms:.4f})"
    
    min_duration = expected_duration * QUALITY_CHECK_PARAMS["MIN_DURATION_FACTOR"]
    max_duration = expected_duration * QUALITY_CHECK_PARAMS["DURATION_TOLERANCE_FACTOR"]
    
    if duration < min_duration:
        return False, f"Audio too short ({duration:.2f}s vs expected {expected_duration:.2f}s)"
    
    if duration > max_duration:
        return False, f"Audio too long ({duration:.2f}s vs expected {expected_duration:.2f}s, tolerance: {max_duration:.2f}s)"
    
    if duration > 5.0:
        segment_length = min(int(sampling_rate * 2), len(audio_array) // 4)
        if segment_length > 0 and len(audio_array) > segment_length * 2:
            first_segment = audio_array[:segment_length]
            mid_segment = audio_array[len(audio_array)//2:len(audio_array)//2 + segment_length]
            
            if len(first_segment) == len(mid_segment):
                correlation = np.corrcoef(first_segment, mid_segment)[0, 1]
                if not np.isnan(correlation) and correlation > QUALITY_CHECK_PARAMS["CORRELATION_THRESHOLD"]:
                    return False, f"Possible repetitive audio detected (correlation: {correlation:.3f})"
    
    return True, f"Quality OK ({duration:.2f}s, max: {audio_max:.3f}, rms: {audio_rms:.3f})"

# ===== LOAD PREDEFINED VOICES =====

def load_voice_profiles():
    """Load voice profiles from YAML files"""
    voice_profiles = {}
    voice_prompts_path = Path(VOICE_PROMPTS_DIR)
        
    if voice_prompts_path.exists():
        for yaml_file in voice_prompts_path.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and 'profiles' in data:
                        voice_profiles.update(data['profiles'])
                        if DEBUG_MODE:
                            print(f"Loaded {len(data['profiles'])} profiles from {yaml_file}")
            except Exception as e:
                print(f"Error loading yaml file {yaml_file}: {e}")
        
    return voice_profiles

def load_predefined_voices():
    """Load available voice prompts from the voice_prompts directory"""
    voices = []
    voice_prompts_path = Path(VOICE_PROMPTS_DIR)
    
    if voice_prompts_path.exists():
        wav_files = list(voice_prompts_path.glob("*.wav"))
        for wav_file in wav_files:
            voice_name = wav_file.stem
            if voice_name.startswith("temp_upload"):
                continue
                
            txt_file = wav_file.with_suffix('.txt')
            if txt_file.exists():
                voices.append(voice_name)
        
        voice_profiles = load_voice_profiles()
        for profile_name in voice_profiles.keys():
            voices.append(f"profile:{profile_name}")
    
    if not voices:
        voices = ["belinda", "broom_salesman", "en_man", "en_woman"]
    
    return sorted(voices)

PREDEFINED_VOICES = load_predefined_voices()

# ===== GENERATION QUEUE =====

class GenerationQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.current_job = None
        self.is_processing = False
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.completed_jobs = {}
        self.status_callback = None
        self.result_callback = None
        
    def add_job(self, job_data):
        job_id = f"job_{int(time.time() * 1000)}"
        job = {
            'id': job_id,
            'data': job_data,
            'status': 'queued',
            'added_time': datetime.now().strftime("%H:%M:%S"),
            'result': None,
            'error': None,
            'audio_path': None,
            'message': None
        }
        self.queue.put(job)
        if self.status_callback:
            self.status_callback()
        return job_id
    
    def get_queue_status(self):
        queue_size = self.queue.qsize()
        current_status = "Idle"
        if self.is_processing and self.current_job:
            current_status = f"Processing: {self.current_job['id']}"
        
        return {
            'queue_size': queue_size,
            'current_status': current_status,
            'completed': self.jobs_completed,
            'failed': self.jobs_failed,
            'current_job': self.current_job
        }
    
    def get_completed_job(self, job_id):
        return self.completed_jobs.get(job_id)
    
    def get_latest_completed_job(self):
        if not self.completed_jobs:
            return None
        
        latest_job = max(self.completed_jobs.values(), 
                        key=lambda x: x.get('completed_time', 0))
        return latest_job
    
    def start_processing(self, higgs_interface):
        """Start the queue processing thread"""
        def process_queue():
            while True:
                try:
                    job = self.queue.get(timeout=1)
                    self.current_job = job
                    self.is_processing = True
                    job['status'] = 'processing'
                    
                    if self.status_callback:
                        self.status_callback()
                    
                    try:
                        class DummyProgress:
                            def __call__(self, progress, desc=""):
                                pass
                        
                        job['data']['progress'] = DummyProgress()
                        
                        audio_path, message = higgs_interface.generate_audio_official(**job['data'])
                        
                        job['audio_path'] = audio_path
                        job['message'] = message
                        job['status'] = 'completed'
                        job['completed_time'] = time.time()
                        
                        self.completed_jobs[job['id']] = job
                        
                        self.jobs_completed += 1
                        
                        if self.result_callback:
                            self.result_callback()
                        
                    except Exception as e:
                        job['error'] = str(e)
                        job['status'] = 'failed'
                        job['completed_time'] = time.time()
                        self.completed_jobs[job['id']] = job
                        self.jobs_failed += 1
                        
                        if self.result_callback:
                            self.result_callback()
                    
                    self.current_job = None
                    self.is_processing = False
                    
                    if self.status_callback:
                        self.status_callback()
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Queue processing error: {e}")
        
        thread = threading.Thread(target=process_queue, daemon=True)
        thread.start()

# ===== HIGGS AUDIO INTERFACE =====

class HiggsAudioInterface:
    def __init__(self):
        self.serve_engine = None
        self.audio_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_queue = GenerationQueue()
        self.load_model()
        self.generation_queue.start_processing(self)
    
    def load_model(self):
        """Load the HiggsAudio model and audio tokenizer"""
        try:
            self.serve_engine = HiggsAudioServeEngine(
                MODEL_PATH, 
                AUDIO_TOKENIZER_PATH, 
                device=self.device
            )
            
            self.audio_tokenizer = load_higgs_audio_tokenizer(
                AUDIO_TOKENIZER_PATH, 
                device=self.device
            )
            
            print(f"Model and audio tokenizer loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def prepare_reference_audio(self, audio_path: str) -> str:
        """Prepare reference audio by saving it to voice_prompts directory if needed"""
        if not audio_path:
            return None
            
        if not audio_path.startswith('/') and not '\\' in audio_path and not audio_path.startswith('C:'):
            return audio_path
            
        temp_name = f"temp_upload_{int(time.time())}"
        dest_audio = Path(VOICE_PROMPTS_DIR) / f"{temp_name}.wav"
        dest_text = Path(VOICE_PROMPTS_DIR) / f"{temp_name}.txt"
        
        shutil.copy2(audio_path, dest_audio)
        
        with open(dest_text, 'w') as f:
            f.write("This is a reference audio sample.")
        
        return temp_name
    
    def debug_print(self, message):
        """Print debug messages if debug mode is enabled"""
        if DEBUG_MODE:
            print(f"[DEBUG] {message}")

    def get_speaker_names_from_job(self, job_data):
        """Extract speaker names from job data for filename generation"""
        speakers = []
        
        if job_data.get('use_reference_audio') and job_data.get('predefined_voices'):
            for voice in job_data['predefined_voices']:
                if voice and voice != "None":
                    if voice.startswith("profile:"):
                        speakers.append(voice.replace("profile:", "prof_"))
                    else:
                        speakers.append(voice)
        
        if job_data.get('use_reference_audio') and job_data.get('reference_audio_files'):
            uploaded_count = 0
            for audio_file in job_data['reference_audio_files']:
                if audio_file is not None:
                    speakers.append(f"upload_{uploaded_count}")
                    uploaded_count += 1
        
        if not speakers:
            speakers = ["default"]
        
        return speakers

    def generate_chunk_with_retries(self, chunk_messages, chunk_text, chunk_idx, chunk_speaker, expected_duration, 
                                   enable_quality_checks, temperature, top_p, top_k, max_new_tokens, seed, 
                                   ras_win_len, ras_win_max_repeat):
        """Generate a single chunk with quality checks and retries"""
        chat_sample = ChatMLSample(messages=chunk_messages)
        
        max_attempts = QUALITY_CHECK_PARAMS["MAX_REGENERATION_ATTEMPTS"] if enable_quality_checks else 1
        
        for attempt in range(max_attempts):
            try:
                if DEBUG_MODE and attempt > 0:
                    print(f"🔄 Retry attempt {attempt + 1}/{max_attempts} for chunk {chunk_idx}")
                
                retry_seed = seed if seed > 0 else None
                if attempt > 0 and retry_seed:
                    retry_seed = retry_seed + attempt * 1000
                
                retry_temperature = temperature
                if attempt > 0:
                    retry_temperature = max(0.1, min(1.0, temperature + (attempt - 1) * 0.1))
                
                chunk_output = self.serve_engine.generate(
                    chat_ml_sample=chat_sample,
                    max_new_tokens=max_new_tokens,
                    temperature=retry_temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                    ras_win_len=ras_win_len if ras_win_len > 0 else None,
                    ras_win_max_num_repeat=ras_win_max_repeat,
                    seed=retry_seed,
                )
                
                if chunk_output.audio is not None:
                    is_good, quality_msg = check_audio_quality(
                        chunk_output.audio, 
                        chunk_output.sampling_rate, 
                        expected_duration, 
                        chunk_text,
                        enable_quality_checks
                    )
                    
                    if is_good:
                        if DEBUG_MODE:
                            if attempt > 0:
                                print(f"✅ Retry successful on attempt {attempt + 1}")
                            print(f"✅ Generated audio: {chunk_output.audio.shape} ({quality_msg})")
                        return chunk_output, True
                    else:
                        if DEBUG_MODE:
                            print(f"❌ Quality check failed (attempt {attempt + 1}): {quality_msg}")
                        
                        if attempt == max_attempts - 1:
                            if DEBUG_MODE:
                                print(f"⚠️ Using potentially problematic audio after {max_attempts} attempts")
                            return chunk_output, False
                        
                        continue
                else:
                    if DEBUG_MODE:
                        print(f"❌ No audio generated (attempt {attempt + 1})")
                    
                    if attempt == max_attempts - 1:
                        return None, False
            
            except Exception as e:
                if DEBUG_MODE:
                    print(f"❌ Generation error (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_attempts - 1:
                    raise e
                
                continue
        
        return None, False

    def generate_audio_official(
        self,
        transcript: str,
        use_reference_audio: bool,
        reference_audio_files: List,
        predefined_voices: List[str],
        scene_prompt_text: str,
        speaker_voice_descriptions: str,
        use_voice_descriptions: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        seed: int,
        ras_win_len: int,
        ras_win_max_repeat: int,
        chunk_method: str,
        chunk_max_word_num: int,
        chunk_max_num_turns: int,
        generation_chunk_buffer_size: int,
        ref_audio_in_system: bool,
        enable_quality_checks: bool = True,
        use_system_placeholder_context: bool = False,
        progress=None
    ) -> Tuple[str, str]:
        """Generate audio using HiggsAudio chunking with system placeholder context"""
        
        if not transcript.strip():
            return None, "Error: Please enter a transcript"
        
        if progress:
            progress(0.1, desc="Preparing generation...")
        
        try:
            if chunk_method == "none" or chunk_method is None:
                return self._generate_simple(
                    transcript, use_reference_audio, reference_audio_files, predefined_voices,
                    scene_prompt_text, speaker_voice_descriptions, use_voice_descriptions,
                    temperature, top_p, top_k, max_new_tokens, seed, ras_win_len, ras_win_max_repeat,
                    ref_audio_in_system, enable_quality_checks, progress
                )
            else:
                # Choose between regular chunking and system placeholder context chunking
                if use_system_placeholder_context:
                    return self._generate_chunked_with_system_placeholder_context(
                        transcript, use_reference_audio, reference_audio_files, predefined_voices,
                        scene_prompt_text, speaker_voice_descriptions, use_voice_descriptions,
                        temperature, top_p, top_k, max_new_tokens, seed, ras_win_len, ras_win_max_repeat,
                        chunk_method, chunk_max_word_num, chunk_max_num_turns, generation_chunk_buffer_size,
                        ref_audio_in_system, enable_quality_checks, progress
                    )
                else:
                    return self._generate_chunked(
                        transcript, use_reference_audio, reference_audio_files, predefined_voices,
                        scene_prompt_text, speaker_voice_descriptions, use_voice_descriptions,
                        temperature, top_p, top_k, max_new_tokens, seed, ras_win_len, ras_win_max_repeat,
                        chunk_method, chunk_max_word_num, chunk_max_num_turns, generation_chunk_buffer_size,
                        ref_audio_in_system, enable_quality_checks, progress
                    )
        except Exception as e:
            import traceback
            error_msg = f"Error generating audio: {str(e)}"
            self.debug_print(f"Generation error: {traceback.format_exc()}")
            return None, error_msg

    def _generate_simple(self, transcript, use_reference_audio, reference_audio_files, predefined_voices,
                        scene_prompt_text, speaker_voice_descriptions, use_voice_descriptions,
                        temperature, top_p, top_k, max_new_tokens, seed, ras_win_len, ras_win_max_repeat,
                        ref_audio_in_system, enable_quality_checks, progress):
        """Generate audio without chunking using serve engine directly"""
        
        if progress:
            progress(0.3, desc="Building messages...")
        
        scene_desc = scene_prompt_text.strip() if scene_prompt_text.strip() else ""
        
        if use_voice_descriptions and speaker_voice_descriptions.strip():
            voice_desc_lines = [line.strip() for line in speaker_voice_descriptions.strip().split('\n') if line.strip()]
            if scene_desc:
                scene_desc += "\n\n" + "\n".join(voice_desc_lines)
            else:
                scene_desc = "\n".join(voice_desc_lines)
        
        if scene_desc:
            system_prompt = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
        else:
            system_prompt = "Generate audio following instruction."
        
        messages = [Message(role="system", content=system_prompt)]
        
        if use_reference_audio:
            ref_audio_names = []
            
            if reference_audio_files:
                for i, audio_file in enumerate(reference_audio_files):
                    if audio_file is not None:
                        ref_name = self.prepare_reference_audio(audio_file)
                        if ref_name:
                            ref_audio_names.append(ref_name)
            
            if predefined_voices:
                for voice in predefined_voices:
                    if voice and voice != "None":
                        ref_audio_names.append(voice)
            
            for i, ref_audio_name in enumerate(ref_audio_names):
                if not ref_audio_name.startswith("profile:"):
                    text_file = Path(VOICE_PROMPTS_DIR) / f"{ref_audio_name}.txt"
                    audio_file = Path(VOICE_PROMPTS_DIR) / f"{ref_audio_name}.wav"
                    
                    if text_file.exists() and audio_file.exists():
                        with open(text_file, 'r') as f:
                            ref_text = f.read().strip()
                        
                        messages.append(Message(role="user", content=ref_text))
                        messages.append(Message(role="assistant", content=AudioContent(audio_url=str(audio_file))))
        
        messages.append(Message(role="user", content=transcript))
        
        if progress:
            progress(0.6, desc="Generating audio...")
        
        expected_duration = estimate_expected_duration(transcript)
        chunk_output, quality_ok = self.generate_chunk_with_retries(
            messages, transcript, 0, None, expected_duration, enable_quality_checks,
            temperature, top_p, top_k, max_new_tokens, seed, ras_win_len, ras_win_max_repeat
        )
        
        if progress:
            progress(0.8, desc="Saving audio...")
        
        if chunk_output is None or chunk_output.audio is None:
            return None, "Error: No audio was generated"
        
        job_data = {
            'use_reference_audio': use_reference_audio,
            'predefined_voices': predefined_voices,
            'reference_audio_files': reference_audio_files
        }
        speaker_names = self.get_speaker_names_from_job(job_data)
        
        if len(speaker_names) == 1:
            speaker_part = speaker_names[0]
        else:
            speaker_part = "_".join(speaker_names[:3])
            if len(speaker_names) > 3:
                speaker_part += f"_plus{len(speaker_names)-3}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        quality_suffix = "_QC" if enable_quality_checks else ""
        filename = f"{speaker_part}_{timestamp}{quality_suffix}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        audio_tensor = torch.from_numpy(chunk_output.audio)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        torchaudio.save(output_path, audio_tensor, chunk_output.sampling_rate)
        
        if progress:
            progress(1.0, desc="Complete!")
        
        result_msg = f"Audio generated successfully! Saved as: {filename}"
        if enable_quality_checks and not quality_ok:
            result_msg += " (Quality issues detected but included)"
        
        return output_path, result_msg

    def _build_chunk_specific_system_message(self, chunk_text, speaker_to_voice, voice_profiles, 
                                            speaker_voice_descriptions_dict, scene_desc, 
                                            speaker_voice_references, all_speaker_tags):
        """Build system message including only speakers in this chunk, following generation.py style exactly"""
        
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        chunk_speakers = sorted(set(pattern.findall(chunk_text)))
        
        if not chunk_speakers:
            if scene_desc:
                return Message(role="system", content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>")
            else:
                return Message(role="system", content="Generate audio following instruction.")
        
        speaker_desc_lines = []
        
        for speaker in chunk_speakers:
            needs_description = True
            speaker_desc = None
            
            if speaker in speaker_voice_references:
                needs_description = False
            
            elif speaker in speaker_to_voice and not speaker_to_voice[speaker].startswith("profile:"):
                voice_name = speaker_to_voice[speaker]
                text_file = Path(VOICE_PROMPTS_DIR) / f"{voice_name}.txt"
                audio_file = Path(VOICE_PROMPTS_DIR) / f"{voice_name}.wav"
                if text_file.exists() and audio_file.exists():
                    needs_description = False
            
            if speaker in speaker_to_voice:
                voice = speaker_to_voice[speaker]
                if voice.startswith("profile:"):
                    needs_description = True
                    profile_name = voice[len("profile:"):].strip()
                    if profile_name in voice_profiles:
                        speaker_desc = voice_profiles[profile_name]
                        if DEBUG_MODE:
                            print(f"Using profile description for {speaker}: {speaker_desc}")
            
            if needs_description:
                if not speaker_desc and speaker in speaker_voice_descriptions_dict:
                    speaker_desc = speaker_voice_descriptions_dict[speaker]
                    if DEBUG_MODE:
                        print(f"Using custom voice description for {speaker}: {speaker_desc}")
                
                if not speaker_desc:
                    try:
                        speaker_num = int(speaker.replace("SPEAKER", ""))
                        speaker_desc = "feminine" if speaker_num % 2 == 0 else "masculine"
                    except:
                        speaker_desc = "neutral voice"
                    if DEBUG_MODE:
                        print(f"Using fallback description for {speaker}: {speaker_desc}")
                
                speaker_desc_lines.append(f"{speaker}: {speaker_desc}")
        
        if speaker_desc_lines:
            scene_parts = []
            if scene_desc:
                scene_parts.append(scene_desc)
            
            if scene_parts:
                scene_content = scene_desc + "\n\n" + "\n".join(speaker_desc_lines)
            else:
                scene_content = "\n".join(speaker_desc_lines)
            
            system_content = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_content}\n<|scene_desc_end|>"
        else:
            if scene_desc:
                system_content = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
            else:
                system_content = "Generate audio following instruction."
        
        if DEBUG_MODE:
            print(f"Chunk-specific system message:")
            print(f"  Chunk speakers: {chunk_speakers}")
            print(f"  Speakers needing descriptions: {[line.split(':')[0] for line in speaker_desc_lines]}")
            print(f"  Profile voices in chunk: {[s for s in chunk_speakers if s in speaker_to_voice and speaker_to_voice[s].startswith('profile:')]}")
            print(f"  Custom descriptions available: {list(speaker_voice_descriptions_dict.keys())}")
            print(f"  System content: {system_content}")
        
        return Message(role="system", content=system_content)

    def _generate_chunked(self, transcript, use_reference_audio, reference_audio_files, predefined_voices,
                         scene_prompt_text, speaker_voice_descriptions, use_voice_descriptions,
                         temperature, top_p, top_k, max_new_tokens, seed, ras_win_len, ras_win_max_repeat,
                         chunk_method, chunk_max_word_num, chunk_max_num_turns, generation_chunk_buffer_size,
                         ref_audio_in_system, enable_quality_checks, progress):
        """Generate audio with chunking and quality control - standard chunking method"""
        
        if progress:
            progress(0.2, desc="Applying text normalization...")
        
        transcript = normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ").replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit")
        transcript = transcript.replace("°C", " degrees Celsius")
        
        sound_effects_map = [
            ("[laugh]", "<SE>[Laughter]</SE>"),
            ("[humming start]", "<SE>[Humming]</SE>"),
            ("[humming end]", "<SE_e>[Humming]</SE_e>"),
            ("[music start]", "<SE_s>[Music]</SE_s>"),
            ("[music end]", "<SE_e>[Music]</SE_e>"),
            ("[music]", "<SE>[Music]</SE>"),
            ("[sing start]", "<SE_s>[Singing]</SE_s>"),
            ("[sing end]", "<SE_e>[Singing]</SE_e>"),
            ("[applause]", "<SE>[Applause]</SE>"),
            ("[cheering]", "<SE>[Cheering]</SE>"),
            ("[cough]", "<SE>[Cough]</SE>"),
        ]
        
        for tag, replacement in sound_effects_map:
            transcript = transcript.replace(tag, replacement)
        
        lines = transcript.split("\n")
        transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
        transcript = transcript.strip()
        
        if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
            transcript += "."
        
        if progress:
            progress(0.3, desc="Chunking text...")
        
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        speaker_tags = sorted(set(pattern.findall(transcript)))
        
        chunked_text = prepare_chunk_text(
            transcript,
            chunk_method=chunk_method,
            chunk_max_word_num=chunk_max_word_num,
            chunk_max_num_turns=chunk_max_num_turns,
        )
        
        processed_chunks = []
        for chunk in chunked_text:
            chunk = chunk.strip()
            if chunk and not any([chunk.endswith(c) for c in [".", "!", "?", "</SE_e>", "</SE>"]]):
                last_punct_pos = max(
                    chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'),
                    chunk.rfind('</SE_e>'), chunk.rfind('</SE>')
                )
                if last_punct_pos > len(chunk) * 0.7:
                    chunk = chunk[:last_punct_pos + 1]
                else:
                    chunk += "."
            processed_chunks.append(chunk)
        
        chunked_text = processed_chunks
        
        if DEBUG_MODE:
            print("Official chunking results:")
            for idx, chunk_text in enumerate(chunked_text):
                print(f"Chunk {idx}: {chunk_text}")
                print("-----")
        
        if progress:
            progress(0.4, desc="Preparing context...")
        
        voice_profiles = load_voice_profiles()
        speaker_to_voice = {}
        ref_audio_names = []
        
        if use_reference_audio:
            if reference_audio_files:
                for audio_file in reference_audio_files:
                    if audio_file is not None:
                        ref_name = self.prepare_reference_audio(audio_file)
                        if ref_name:
                            ref_audio_names.append(ref_name)
            
            if predefined_voices:
                for voice in predefined_voices:
                    if voice and voice != "None":
                        ref_audio_names.append(voice)
            
            for i, speaker in enumerate(speaker_tags):
                if i < len(ref_audio_names):
                    speaker_to_voice[speaker] = ref_audio_names[i]
                elif ref_audio_names:
                    speaker_to_voice[speaker] = ref_audio_names[i % len(ref_audio_names)]
        
        has_profile_voices = False
        if speaker_to_voice:
            for speaker, voice in speaker_to_voice.items():
                if voice.startswith("profile:"):
                    has_profile_voices = True
                    break
        
        if has_profile_voices:
            ref_audio_in_system = True
            if DEBUG_MODE:
                print(f"🎯 DETECTED PROFILE VOICES - forcing ref_audio_in_system=True")
                print(f"Profile voices: {[f'{s}:{v}' for s, v in speaker_to_voice.items() if v.startswith('profile:')]}")
        
        speaker_voice_descriptions_dict = {}
        if use_voice_descriptions and speaker_voice_descriptions.strip():
            voice_desc_lines = [line.strip() for line in speaker_voice_descriptions.strip().split('\n') if line.strip()]
            for line in voice_desc_lines:
                if ':' in line:
                    speaker_id, description = line.split(':', 1)
                    speaker_voice_descriptions_dict[speaker_id.strip()] = description.strip()
            
            if DEBUG_MODE:
                print(f"Parsed voice descriptions: {speaker_voice_descriptions_dict}")
        
        if DEBUG_MODE:
            print(f"Speaker to voice mapping: {speaker_to_voice}")
            print(f"Available voices: {ref_audio_names}")
            print(f"Loaded voice profiles: {list(voice_profiles.keys())}")
            print(f"Using ref_audio_in_system: {ref_audio_in_system}")
            print(f"Has profile voices: {has_profile_voices}")
            print(f"Voice descriptions enabled: {use_voice_descriptions}")
            print(f"Voice descriptions dict: {speaker_voice_descriptions_dict}")
            print(f"Quality checks enabled: {enable_quality_checks}")
            print(f"Total chunks to generate: {len(chunked_text)}")
            print(f"Max tokens per chunk: {max_new_tokens}")
        
        speaker_voice_references = {}
        speakers_needing_voice_establishment = []
        for speaker in speaker_tags:
            if speaker not in speaker_to_voice:
                speakers_needing_voice_establishment.append(speaker)
            elif speaker in speaker_to_voice:
                voice = speaker_to_voice[speaker]
                if voice.startswith("profile:"):
                    speakers_needing_voice_establishment.append(speaker)
        
        if DEBUG_MODE:
            print(f"Speakers needing voice establishment: {speakers_needing_voice_establishment}")
            print(f"Speakers with predefined audio files: {[s for s in speaker_tags if s in speaker_to_voice and not speaker_to_voice[s].startswith('profile:')]}")
        
        all_audio_segments = []
        total_chunks = len(chunked_text)
        quality_issues_count = 0
        
        for idx, chunk_text in enumerate(chunked_text):
            if progress:
                chunk_progress = 0.5 + (0.4 * (idx + 1) / total_chunks)
                progress(chunk_progress, desc=f"Processing chunk {idx + 1}/{total_chunks}")
            
            chunk_speaker = None
            chunk_speaker_match = re.search(r'\[(SPEAKER\d+)\]', chunk_text)
            if chunk_speaker_match:
                chunk_speaker = chunk_speaker_match.group(1)
            
            expected_duration = estimate_expected_duration(chunk_text)
            
            chunk_messages = []
            voice_strategy = "unknown"
            
            if chunk_speaker:
                if chunk_speaker in speaker_voice_references:
                    ref_audio_path, ref_text = speaker_voice_references[chunk_speaker]
                    
                    scene_desc = scene_prompt_text.strip() if scene_prompt_text.strip() else ""
                    if scene_desc:
                        ref_system_content = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
                    else:
                        ref_system_content = "Generate audio following instruction."
                    
                    chunk_messages.append(Message(role="system", content=ref_system_content))
                    chunk_messages.append(Message(role="user", content=ref_text))
                    chunk_messages.append(Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)))
                    
                    voice_strategy = f"established_reference"
                
                elif chunk_speaker in speaker_to_voice and not speaker_to_voice[chunk_speaker].startswith("profile:"):
                    voice_name = speaker_to_voice[chunk_speaker]
                    text_file = Path(VOICE_PROMPTS_DIR) / f"{voice_name}.txt"
                    audio_file = Path(VOICE_PROMPTS_DIR) / f"{voice_name}.wav"
                    
                    if text_file.exists() and audio_file.exists():
                        with open(text_file, 'r') as f:
                            ref_text = f.read().strip()
                        
                        scene_desc = scene_prompt_text.strip() if scene_prompt_text.strip() else ""
                        if scene_desc:
                            ref_system_content = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
                        else:
                            ref_system_content = "Generate audio following instruction."
                        
                        chunk_messages.append(Message(role="system", content=ref_system_content))
                        
                        if len(speaker_tags) > 1:
                            ref_text = f"[{chunk_speaker}] {ref_text}"
                        
                        chunk_messages.append(Message(role="user", content=ref_text))
                        chunk_messages.append(Message(role="assistant", content=AudioContent(audio_url=str(audio_file))))
                        
                        voice_strategy = f"predefined_audio: {voice_name}"
                    else:
                        chunk_system = self._build_chunk_specific_system_message(
                            chunk_text, speaker_to_voice, voice_profiles, 
                            speaker_voice_descriptions_dict, scene_prompt_text.strip(), 
                            speaker_voice_references, speaker_tags
                        )
                        chunk_messages.append(chunk_system)
                        voice_strategy = "voice_descriptions_fallback"
                
                else:
                    chunk_system = self._build_chunk_specific_system_message(
                        chunk_text, speaker_to_voice, voice_profiles, 
                        speaker_voice_descriptions_dict, scene_prompt_text.strip(), 
                        speaker_voice_references, speaker_tags
                    )
                    chunk_messages.append(chunk_system)
                    voice_strategy = "first_generation_with_profiles" if has_profile_voices else "first_generation"
            else:
                chunk_system = self._build_chunk_specific_system_message(
                    chunk_text, speaker_to_voice, voice_profiles, 
                    speaker_voice_descriptions_dict, scene_prompt_text.strip(), 
                    speaker_voice_references, speaker_tags
                )
                chunk_messages.append(chunk_system)
                voice_strategy = "no_speaker"
            
            chunk_messages.append(Message(role="user", content=chunk_text))
            
            if DEBUG_MODE:
                print(f"\n--- Generating Chunk {idx} ---")
                print(f"Speaker: {chunk_speaker}")
                print(f"Voice strategy: {voice_strategy}")
                print(f"Text: {chunk_text}")
                print(f"Expected duration: {expected_duration:.2f}s")
                print(f"Text length: {len(chunk_text.strip())} chars")
                print(f"Tokens available: {max_new_tokens}")
                print(f"Messages in context: {len(chunk_messages)}")
                print(f"Quality checks enabled: {enable_quality_checks}")
            
            try:
                chunk_output, quality_ok = self.generate_chunk_with_retries(
                    chunk_messages, chunk_text, idx, chunk_speaker, expected_duration, enable_quality_checks,
                    temperature, top_p, top_k, max_new_tokens, seed, ras_win_len, ras_win_max_repeat
                )
                
                if chunk_output and chunk_output.audio is not None:
                    all_audio_segments.append(chunk_output.audio)
                    
                    if not quality_ok:
                        quality_issues_count += 1
                    
                    if (chunk_speaker and 
                        chunk_speaker in speakers_needing_voice_establishment and 
                        chunk_speaker not in speaker_voice_references and
                        voice_strategy in ["first_generation", "voice_descriptions_fallback", "first_generation_with_profiles"] and
                        not (chunk_speaker in speaker_to_voice and speaker_to_voice[chunk_speaker].startswith("profile:"))):
                        
                        duration = len(chunk_output.audio) / chunk_output.sampling_rate
                        audio_max = np.max(np.abs(chunk_output.audio))
                        audio_rms = np.sqrt(np.mean(chunk_output.audio ** 2))
                        
                        if quality_ok and (audio_max >= 0.01 and audio_rms >= 0.003 and 
                                         duration >= 2.0 and duration <= 40.0):
                            
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                temp_path = temp_file.name
                                audio_tensor = torch.from_numpy(chunk_output.audio)
                                if audio_tensor.dim() == 1:
                                    audio_tensor = audio_tensor.unsqueeze(0)
                                torchaudio.save(temp_path, audio_tensor, chunk_output.sampling_rate)
                            
                            speaker_voice_references[chunk_speaker] = (temp_path, chunk_text)
                            
                            if DEBUG_MODE:
                                print(f"🎯 ESTABLISHED HIGH-QUALITY VOICE REFERENCE for {chunk_speaker}")
                else:
                    if DEBUG_MODE:
                        print(f"❌ Failed to generate chunk {idx} after all attempts")
                    continue
                    
            except Exception as e:
                if DEBUG_MODE:
                    print(f"❌ Critical error generating chunk {idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
        
        if progress:
            progress(0.9, desc="Concatenating audio...")
        
        if not all_audio_segments:
            return None, "Error: No audio was generated from any chunks"
        
        concat_audio = np.concatenate(all_audio_segments)
        
        job_data = {
            'use_reference_audio': use_reference_audio,
            'predefined_voices': predefined_voices,
            'reference_audio_files': reference_audio_files
        }
        speaker_names = self.get_speaker_names_from_job(job_data)
        
        if len(speaker_names) == 1:
            speaker_part = speaker_names[0]
        else:
            speaker_part = "_".join(speaker_names[:3])
            if len(speaker_names) > 3:
                speaker_part += f"_plus{len(speaker_names)-3}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        quality_suffix = "_QC" if enable_quality_checks else ""
        filename = f"{speaker_part}_chunked_{chunk_method}_{timestamp}{quality_suffix}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        audio_tensor = torch.from_numpy(concat_audio)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        sampling_rate = getattr(chunk_output, 'sampling_rate', 24000)
        torchaudio.save(output_path, audio_tensor, sampling_rate)
        
        if progress:
            progress(1.0, desc="Complete!")
        
        for speaker, (temp_path, _) in speaker_voice_references.items():
            try:
                os.unlink(temp_path)
                if DEBUG_MODE:
                    print(f"Cleaned up temporary reference file for {speaker}: {temp_path}")
            except:
                pass
        
        if DEBUG_MODE:
            print(f"\n=== GENERATION COMPLETE ===")
            print(f"Generated chunks: {len(all_audio_segments)}")
            print(f"Quality issues encountered: {quality_issues_count}")
            print(f"Total audio samples: {len(concat_audio)}")
            print(f"Total duration: {len(concat_audio) / sampling_rate:.2f} seconds")
            print(f"Chunking method: {chunk_method}")
            print(f"Tokens per chunk: {max_new_tokens}")
            print(f"Quality checks enabled: {enable_quality_checks}")
            print(f"Output file: {filename}")
        
        final_message = f"Audio generated successfully using {chunk_method} chunking!"
        if enable_quality_checks and quality_issues_count > 0:
            final_message += f" Note: {quality_issues_count} chunks had quality issues but were included."
        final_message += f" Saved as: {filename}"
        
        return output_path, final_message

    def _generate_chunked_with_system_placeholder_context(self, transcript, use_reference_audio, reference_audio_files, predefined_voices,
                                                           scene_prompt_text, speaker_voice_descriptions, use_voice_descriptions,
                                                           temperature, top_p, top_k, max_new_tokens, seed, ras_win_len, ras_win_max_repeat,
                                                           chunk_method, chunk_max_word_num, chunk_max_num_turns, generation_chunk_buffer_size,
                                                           ref_audio_in_system, enable_quality_checks, progress):
        
        if progress:
            progress(0.2, desc="Applying text normalization...")
        
        # Apply text normalization
        transcript = normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ").replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit")
        transcript = transcript.replace("°C", " degrees Celsius")
        
        # Apply sound effect transformations
        sound_effects_map = [
            ("[laugh]", "<SE>[Laughter]</SE>"),
            ("[humming start]", "<SE>[Humming]</SE>"),
            ("[humming end]", "<SE_e>[Humming]</SE_e>"),
            ("[music start]", "<SE_s>[Music]</SE_s>"),
            ("[music end]", "<SE_e>[Music]</SE_e>"),
            ("[music]", "<SE>[Music]</SE>"),
            ("[sing start]", "<SE_s>[Singing]</SE_s>"),
            ("[sing end]", "<SE_e>[Singing]</SE_e>"),
            ("[applause]", "<SE>[Applause]</SE>"),
            ("[cheering]", "<SE>[Cheering]</SE>"),
            ("[cough]", "<SE>[Cough]</SE>"),
        ]
        
        for tag, replacement in sound_effects_map:
            transcript = transcript.replace(tag, replacement)
        
        # Clean up text and ensure proper sentence endings
        lines = transcript.split("\n")
        transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
        transcript = transcript.strip()
        
        if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
            transcript += "."
        
        if progress:
            progress(0.3, desc="Chunking text...")
        
        # Extract speaker tags and chunk text
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        speaker_tags = sorted(set(pattern.findall(transcript)))
        
        chunked_text = prepare_chunk_text(
            transcript,
            chunk_method=chunk_method,
            chunk_max_word_num=chunk_max_word_num,
            chunk_max_num_turns=chunk_max_num_turns,
        )
        
        # Post-process chunks
        processed_chunks = []
        for chunk in chunked_text:
            chunk = chunk.strip()
            if chunk and not any([chunk.endswith(c) for c in [".", "!", "?", "</SE_e>", "</SE>"]]):
                last_punct_pos = max(
                    chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'),
                    chunk.rfind('</SE_e>'), chunk.rfind('</SE>')
                )
                if last_punct_pos > len(chunk) * 0.7:
                    chunk = chunk[:last_punct_pos + 1]
                else:
                    chunk += "."
            processed_chunks.append(chunk)
        
        chunked_text = processed_chunks
        
        if progress:
            progress(0.4, desc="Preparing voice baseline context...")
        
        # Build initial reference audio mapping
        initial_ref_audio_names = []
        speaker_to_ref_audio = {}
        
        if use_reference_audio:
            # Process uploaded files
            if reference_audio_files:
                for audio_file in reference_audio_files:
                    if audio_file is not None:
                        ref_name = self.prepare_reference_audio(audio_file)
                        if ref_name:
                            initial_ref_audio_names.append(ref_name)
            
            # Process predefined voices
            if predefined_voices:
                for voice in predefined_voices:
                    if voice and voice != "None":
                        initial_ref_audio_names.append(voice)
            
            # Map speakers to reference audio
            for i, speaker in enumerate(speaker_tags):
                if i < len(initial_ref_audio_names):
                    speaker_to_ref_audio[speaker] = initial_ref_audio_names[i]
                elif initial_ref_audio_names:
                    speaker_to_ref_audio[speaker] = initial_ref_audio_names[i % len(initial_ref_audio_names)]
        
        # Parse voice descriptions
        speaker_voice_descriptions_dict = {}
        if use_voice_descriptions and speaker_voice_descriptions.strip():
            voice_desc_lines = [line.strip() for line in speaker_voice_descriptions.strip().split('\n') if line.strip()]
            for line in voice_desc_lines:
                if ':' in line:
                    speaker_id, description = line.split(':', 1)
                    speaker_voice_descriptions_dict[speaker_id.strip()] = description.strip()
        
        # Basic quality check for voice baseline establishment
        def is_suitable_for_voice_baseline(audio_array, sampling_rate, text_content):
            """Check if audio is suitable to become a voice baseline - be pragmatic, not perfectionist"""
            if audio_array is None or len(audio_array) == 0:
                return False, "No audio"
            
            duration = len(audio_array) / sampling_rate
            audio_rms = np.sqrt(np.mean(audio_array ** 2))
            audio_max = np.max(np.abs(audio_array))
            
            # PRAGMATIC requirements for voice baseline: just ensure it's usable audio
            if duration < 5.0:  # At least some substantial audio
                return False, f"Too short ({duration:.1f}s)"
            if duration > 35.01:  # Allow up to 2 minutes - long audio is GOOD for voice learning!
                return False, f"Extremely long ({duration:.1f}s)"
            if audio_rms < 0.003:  # Very quiet threshold - most real speech will pass
                return False, f"Too quiet (RMS: {audio_rms:.4f})"
            #if audio_max > 0.999:  # Only reject severe clipping
            #    return False, f"Severely clipped (max: {audio_max:.3f})"
            
            return True, f"Voice baseline suitable ({duration:.1f}s, RMS: {audio_rms:.3f})"
        
        if DEBUG_MODE:
            print(f"=== VOICE BASELINE + ROLLING CONTEXT SETUP ===")
            print(f"Initial reference audio: {initial_ref_audio_names}")
            print(f"Speaker to reference audio mapping: {speaker_to_ref_audio}")
            print(f"Speaker tags: {speaker_tags}")
            print(f"Chunks to process: {len(chunked_text)}")
            print(f"Buffer size: {generation_chunk_buffer_size}")
            print(f"Strategy: Voice baseline (first best) + rolling context (recent N)")
        
        # Initialize storage systems
        all_audio_segments = []
        
        # VOICE BASELINE: First high-quality generation per speaker (NEVER changes)
        speaker_voice_baseline = {}  # Dict[speaker] -> (text, audio_path, sampling_rate, quality_score)
        
        # ROLLING CONTEXT: Recent N generations per speaker (sliding window)
        speaker_rolling_context = {}  # Dict[speaker] -> List[(text, audio_path, sampling_rate)]
        
        quality_issues_count = 0
        total_chunks = len(chunked_text)
        
        # Initialize storage for each speaker
        for speaker in speaker_tags:
            speaker_voice_baseline[speaker] = None
            speaker_rolling_context[speaker] = []
        
        for idx, chunk_text in enumerate(chunked_text):
            if progress:
                chunk_progress = 0.5 + (0.4 * (idx + 1) / total_chunks)
                progress(chunk_progress, desc=f"Processing chunk {idx + 1}/{total_chunks} (Voice Baseline + Context)")
            
            # Identify speakers in this chunk
            chunk_speaker_matches = re.findall(r'\[(SPEAKER\d+)\]', chunk_text)
            primary_speaker = chunk_speaker_matches[0] if chunk_speaker_matches else None
            chunk_speakers = set(chunk_speaker_matches)
            
            # Build messages for this chunk
            chunk_messages = []
            
            # Build system message with scene description
            scene_desc_parts = []
            if scene_prompt_text.strip():
                scene_desc_parts.append(scene_prompt_text.strip())
            
            # Add voice descriptions for speakers without any reference (original or baseline)
            speakers_needing_description = []
            for speaker in chunk_speakers:
                has_original_ref = speaker in speaker_to_ref_audio
                has_voice_baseline = speaker_voice_baseline[speaker] is not None
                if not has_original_ref and not has_voice_baseline:
                    speakers_needing_description.append(speaker)
            
            if use_voice_descriptions and speaker_voice_descriptions_dict and speakers_needing_description:
                speaker_desc_lines = []
                for speaker in sorted(speakers_needing_description):
                    if speaker in speaker_voice_descriptions_dict:
                        speaker_desc = speaker_voice_descriptions_dict[speaker]
                    else:
                        try:
                            speaker_num = int(speaker.replace("SPEAKER", ""))
                            speaker_desc = "feminine" if speaker_num % 2 == 0 else "masculine"
                        except:
                            speaker_desc = "neutral voice"
                    speaker_desc_lines.append(f"{speaker}: {speaker_desc}")
                if speaker_desc_lines:
                    scene_desc_parts.append("\n".join(speaker_desc_lines))
            
            # Build system message
            if scene_desc_parts:
                system_content = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n" + "\n\n".join(scene_desc_parts) + "\n<|scene_desc_end|>"
            else:
                system_content = "Generate audio following instruction."
            
            chunk_messages.append(Message(role="system", content=system_content))
            
            # PRIORITY 1: Original reference audio (highest priority - never degrades)
            original_ref_used = []
            for speaker in sorted(chunk_speakers):
                if speaker in speaker_to_ref_audio:
                    ref_audio_name = speaker_to_ref_audio[speaker]
                    if not ref_audio_name.startswith("profile:"):
                        text_file = Path(VOICE_PROMPTS_DIR) / f"{ref_audio_name}.txt"
                        audio_file = Path(VOICE_PROMPTS_DIR) / f"{ref_audio_name}.wav"
                        
                        if text_file.exists() and audio_file.exists():
                            with open(text_file, 'r') as f:
                                ref_text = f.read().strip()
                            
                            # Add speaker tag for multi-speaker scenarios
                            if len(speaker_tags) > 1:
                                ref_text = f"[{speaker}] {ref_text}"
                            
                            chunk_messages.append(Message(role="user", content=ref_text))
                            chunk_messages.append(Message(role="assistant", content=AudioContent(audio_url=str(audio_file))))
                            original_ref_used.append(speaker)
            
            # PRIORITY 2: Voice baseline (first best generation - prevents degradation)
            voice_baseline_used = []
            for speaker in sorted(chunk_speakers):
                if speaker not in original_ref_used and speaker_voice_baseline[speaker] is not None:
                    baseline_text, baseline_path, baseline_sr, baseline_quality = speaker_voice_baseline[speaker]
                    if os.path.exists(baseline_path):
                        chunk_messages.append(Message(role="user", content=baseline_text))
                        chunk_messages.append(Message(role="assistant", content=AudioContent(audio_url=baseline_path)))
                        voice_baseline_used.append(speaker)
            
            # PRIORITY 3: Rolling context (recent conversation for flow)
            rolling_context_used = []
            for speaker in sorted(chunk_speakers):
                if speaker not in original_ref_used and speaker in speaker_rolling_context:
                    available_context = speaker_rolling_context[speaker]
                    
                    if available_context and generation_chunk_buffer_size > 0:
                        # Take the most recent N examples (but limit to prevent context overflow)
                        max_recent = min(3, generation_chunk_buffer_size, len(available_context))
                        recent_context = available_context[-max_recent:]
                        
                        for ctx_text, ctx_audio_path, ctx_sampling_rate in recent_context:
                            if os.path.exists(ctx_audio_path):
                                chunk_messages.append(Message(role="user", content=ctx_text))
                                chunk_messages.append(Message(role="assistant", content=AudioContent(audio_url=ctx_audio_path)))
                                rolling_context_used.append(f"{speaker}(recent)")
            
            # Add current chunk to generate
            chunk_messages.append(Message(role="user", content=chunk_text))
            
            if DEBUG_MODE:
                print(f"\n--- Generating Chunk {idx} with VOICE BASELINE + ROLLING CONTEXT ---")
                print(f"Text: {chunk_text}")
                print(f"Primary speaker: {primary_speaker}")
                print(f"Speakers in chunk: {sorted(chunk_speakers)}")
                print(f"Original reference used: {original_ref_used}")
                print(f"Voice baseline used: {voice_baseline_used}")
                print(f"Rolling context used: {rolling_context_used}")
                print(f"Total messages in context: {len(chunk_messages)}")
                
                # Show current state for each speaker
                for speaker in speaker_tags:
                    has_original = speaker in speaker_to_ref_audio
                    has_baseline = speaker_voice_baseline[speaker] is not None
                    rolling_count = len(speaker_rolling_context[speaker])
                    baseline_quality = speaker_voice_baseline[speaker][3] if has_baseline else "N/A"
                    print(f"  {speaker}: Original={has_original}, Baseline={has_baseline}(Q:{baseline_quality}), Rolling={rolling_count}")
            
            # Generate chunk
            try:
                expected_duration = estimate_expected_duration(chunk_text)
                
                # Generate with retries
                chunk_output, quality_ok = self.generate_chunk_with_retries(
                    chunk_messages, chunk_text, idx, primary_speaker, expected_duration, enable_quality_checks,
                    temperature, top_p, top_k, max_new_tokens, seed, ras_win_len, ras_win_max_repeat
                )
                
                if chunk_output and chunk_output.audio is not None:
                    # CRITICAL: Add to output FIRST
                    all_audio_segments.append(chunk_output.audio)
                    
                    if not quality_ok:
                        quality_issues_count += 1
                    
                    # VOICE BASELINE + ROLLING CONTEXT STORAGE
                    sampling_rate = getattr(chunk_output, 'sampling_rate', 24000)
                    
                    # Save this chunk for context (both baseline and rolling possibilities)
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=OUTPUT_DIR) as temp_file:
                            temp_path = temp_file.name
                        
                        # Save audio
                        audio_tensor = torch.from_numpy(chunk_output.audio)
                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        
                        torchaudio.save(temp_path, audio_tensor, sampling_rate)
                        
                        # Verify file was saved
                        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                            # Process each speaker that appeared in this chunk
                            for speaker in chunk_speakers:
                                # VOICE BASELINE LOGIC: Establish if this speaker doesn't have one yet
                                if speaker_voice_baseline[speaker] is None:
                                    is_baseline_worthy, baseline_msg = is_suitable_for_voice_baseline(
                                        chunk_output.audio, sampling_rate, chunk_text
                                    )
                                    
                                    if is_baseline_worthy:
                                        # This becomes the voice baseline for this speaker!
                                        quality_score = 0.9 if quality_ok else 0.7
                                        speaker_voice_baseline[speaker] = (chunk_text, temp_path, sampling_rate, quality_score)
                                        
                                        if DEBUG_MODE:
                                            print(f"🎯 ESTABLISHED VOICE BASELINE for {speaker}: {baseline_msg}")
                                            print(f"   Text: {chunk_text[:50]}...")
                                            print(f"   File: {temp_path}")
                                    else:
                                        if DEBUG_MODE:
                                            print(f"⚠️ Not suitable for voice baseline ({speaker}): {baseline_msg}")
                                
                                # ROLLING CONTEXT LOGIC: Always add to rolling context
                                speaker_rolling_context[speaker].append((chunk_text, temp_path, sampling_rate))
                                
                                # Keep rolling context size manageable
                                max_rolling_per_speaker = 1 # Keep last 2 examples
                                if len(speaker_rolling_context[speaker]) > max_rolling_per_speaker:
                                    # Remove oldest from rolling context (but don't delete file if it's the voice baseline)
                                    old_text, old_path, old_sr = speaker_rolling_context[speaker].pop(0)
                                    
                                    # Only delete if it's not the voice baseline
                                    is_voice_baseline = (speaker_voice_baseline[speaker] is not None and 
                                                       speaker_voice_baseline[speaker][1] == old_path)
                                    if not is_voice_baseline:
                                        try:
                                            if os.path.exists(old_path):
                                                os.unlink(old_path)
                                        except:
                                            pass
                            
                            if DEBUG_MODE:
                                duration = len(chunk_output.audio) / sampling_rate
                                file_size = os.path.getsize(temp_path)
                                #print(f"✅ Generated audio: {chunk_output.audio.shape} ({duration:.2f}s)")
                                print(f"📊 Updated context for speakers: {sorted(chunk_speakers)}")
                                print(f"💾 Saved: {temp_path} ({file_size} bytes)")
                        else:
                            if DEBUG_MODE:
                                print(f"❌ Failed to save context file: {temp_path}")
                            
                    except Exception as context_save_error:
                        if DEBUG_MODE:
                            print(f"⚠️ Context save failed but audio preserved: {context_save_error}")
                        
                else:
                    if DEBUG_MODE:
                        print(f"❌ Failed to generate chunk {idx}")
                    continue
                    
            except Exception as e:
                if DEBUG_MODE:
                    print(f"❌ Error generating chunk {idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
        
        if progress:
            progress(0.9, desc="Concatenating audio...")
        
        # Clean up context files (but preserve voice baselines)
        for speaker in speaker_tags:
            # Clean up rolling context
            for _, temp_path, _ in speaker_rolling_context[speaker]:
                # Only delete if it's not a voice baseline
                is_voice_baseline = (speaker_voice_baseline[speaker] is not None and 
                                  speaker_voice_baseline[speaker][1] == temp_path)
                if not is_voice_baseline:
                    try:
                        if temp_path and os.path.exists(temp_path):
                            os.unlink(temp_path)
                            if DEBUG_MODE:
                                print(f"🧹 Cleaned up rolling context: {temp_path}")
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"⚠️ Cleanup failed for {temp_path}: {e}")
            
            # Clean up voice baseline
            if speaker_voice_baseline[speaker] is not None:
                _, baseline_path, _, _ = speaker_voice_baseline[speaker]
                try:
                    if baseline_path and os.path.exists(baseline_path):
                        os.unlink(baseline_path)
                        if DEBUG_MODE:
                            print(f"🧹 Cleaned up voice baseline: {baseline_path}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"⚠️ Voice baseline cleanup failed: {e}")
        
        if not all_audio_segments:
            return None, "Error: No audio was generated from any chunks"
        
        if DEBUG_MODE:
            print(f"\n=== VOICE BASELINE + ROLLING CONTEXT SUMMARY ===")
            print(f"Total segments: {len(all_audio_segments)}")
            print(f"Quality issues: {quality_issues_count}")
            for speaker in speaker_tags:
                has_baseline = speaker_voice_baseline[speaker] is not None
                rolling_count = len(speaker_rolling_context[speaker])
                baseline_quality = speaker_voice_baseline[speaker][3] if has_baseline else "N/A"
                print(f"{speaker}: Voice baseline={has_baseline}(Q:{baseline_quality}), Rolling context={rolling_count}")
        
        # Concatenate and save
        concat_audio = np.concatenate(all_audio_segments)
        
        # Generate filename  
        job_data = {
            'use_reference_audio': use_reference_audio,
            'predefined_voices': predefined_voices,
            'reference_audio_files': reference_audio_files
        }
        speaker_names = self.get_speaker_names_from_job(job_data)
        
        if len(speaker_names) == 1:
            speaker_part = speaker_names[0]
        else:
            speaker_part = "_".join(speaker_names[:3])
            if len(speaker_names) > 3:
                speaker_part += f"_plus{len(speaker_names)-3}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        quality_suffix = "_QC" if enable_quality_checks else ""
        context_suffix = "_BASELINE"  
        filename = f"{speaker_part}_chunked_{chunk_method}_{timestamp}{quality_suffix}{context_suffix}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save final audio
        audio_tensor = torch.from_numpy(concat_audio)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        final_sampling_rate = getattr(chunk_output, 'sampling_rate', 24000) if 'chunk_output' in locals() else 24000
        torchaudio.save(output_path, audio_tensor, final_sampling_rate)
        
        if progress:
            progress(1.0, desc="Complete!")
        
        if DEBUG_MODE:
            print(f"\n=== VOICE BASELINE + ROLLING CONTEXT GENERATION COMPLETE ===")
            print(f"Generated chunks: {len(all_audio_segments)}")
            print(f"Quality issues: {quality_issues_count}")
            print(f"Total duration: {len(concat_audio) / final_sampling_rate:.2f} seconds")
            print(f"Voice baselines established: {sum(1 for s in speaker_tags if speaker_voice_baseline[s] is not None)}/{len(speaker_tags)}")
            print(f"Output: {filename}")
        
        final_message = f"Voice baseline + rolling context audio generated successfully using {chunk_method} chunking!"
        if enable_quality_checks and quality_issues_count > 0:
            final_message += f" Note: {quality_issues_count} chunks had quality issues but were included."
        final_message += f" Voice stability maintained via baseline references with conversation flow. Saved as: {filename}"
        
        return output_path, final_message    
# Initialize the interface
higgs_interface = HiggsAudioInterface()

# ===== HELP CONTENT =====

def create_help_content():
    """Create comprehensive help content with examples"""
    help_content = """
# HiggsAudio Generation Help (Official Implementation with Quality Control + System Placeholder Context)

## NEW: System Placeholder Context Feature

### What is System Placeholder Context?
This advanced feature uses HiggsAudio's native **AUDIO_PLACEHOLDER_TOKEN** system to inject previous audio chunks directly into the system message of subsequent generations. This maintains superior cross-chunk consistency and speaker voice coherence.

### When to Use System Placeholder Context
- **Multi-speaker conversations**: Maintains distinct speaker voices across long dialogs
- **Narrative content**: Preserves narrator voice consistency
- **Complex audio**: When voice consistency is critical
- **Long-form generation**: Better than standard chunking for maintaining coherence

### How It Works
1. Each generated chunk is temporarily saved
2. Recent chunks are injected into the system message as audio placeholders
3. The model uses these audio references to maintain voice consistency
4. Buffer size controls how many previous chunks to include

## Quality Control Features

### Automatic Quality Checks
- **Duration validation**: Detects audio that's too long or short compared to text length
- **Silence detection**: Identifies very quiet or silent audio
- **Repetition detection**: Finds potentially repetitive/looping audio patterns
- **Automatic retry**: Regenerates problematic chunks up to 3 times with parameter variations

### Quality Control Toggle
- **Enable Quality Checks**: Automatic detection and regeneration of poor-quality chunks
- **Disable Quality Checks**: Accepts all generated audio (faster but may include issues)
- Quality issues are logged in debug mode for analysis

## Transcript Format

### Single Speaker
For single-speaker audio, simply enter your text:
```
Hey, everyone! Welcome back to Tech Talk Tuesdays.
It's your host, Alex, and today, we're diving into deep learning.
```

### Multi-Speaker Dialog
For conversations, use speaker tags:
```
[SPEAKER0] I can't believe you did that without asking me!
[SPEAKER1] Oh, come on! It wasn't a big deal.
[SPEAKER0] You made a decision that affects both of us!
```

### Experimental Features
You can try these experimental formats:

**Humming:**
```
Are you asking if I can hum a tune? Of course I can! [humming start] la la la la la [humming end] See?
```

**Background Music:**
```
[music start] I will remember this, thought Ender, when I am defeated. [music end]
```

## Official Chunking Methods

This interface uses the two approaches:

### Standard Chunking (Default)
- Uses reference audio approach
- Establishes speaker voices through initial examples
- Maintains consistency through voice reference storage

### System Placeholder Context 
- Injects recent audio directly into system messages
- Uses AUDIO_PLACEHOLDER_TOKEN support
- Superior cross-chunk consistency
- Higher memory usage but better quality

### Chunking Options

#### No Chunking (None)
- Processes entire text at once
- Best for shorter texts
- Fastest generation

#### Speaker Chunking
- Splits text by speaker tags `[SPEAKER0]`, `[SPEAKER1]`, etc.
- Processes each speaker turn separately
- **chunk_max_num_turns**: Controls how many speaker turns to group together
- Ideal for multi-speaker conversations

#### Word Chunking  
- Splits text by word count
- **chunk_max_word_num**: Maximum words per chunk (default: 200)
- Automatically detects language (Chinese vs English)
- Best for long-form single-speaker content
- Uses **generation_chunk_buffer_size** to control memory usage

## Voice Options

### Multiple Reference Audio Files
- Upload multiple audio files (one for each speaker)
- Each file will be assigned to SPEAKER0, SPEAKER1, etc. in order

### Multiple Predefined Voices
- Select multiple voices from the dropdown
- Hold Ctrl/Cmd to select multiple options
- Supports profile voices like `profile:male_en_british`

### Voice Descriptions
- Enable "Use Voice Descriptions" to add custom voice characteristics
- Format: `SPEAKER0: masculine; deep voice; british accent`
- Format: `SPEAKER1: feminine; high pitch; excited tone`

## Generation Settings

### Context Control (NEW)
- **Use System Placeholder Context**: Enable advanced context-aware generation
- **Generation Buffer Size**: Controls how many previous chunks to include in context
- **Reference Audio in System**: Includes ref audio in system message

### Quality Control
- **Enable Quality Checks**: Automatic quality control and regeneration
- **Chunk Method**: Uses HiggsAudio chunking (none, speaker, word)
- **Chunk Max Words**: For word chunking (50-500, default 200)
- **Chunk Max Turns**: For speaker chunking (1-10, default 1)

### Advanced Parameters
- **Temperature**: Controls randomness (0.1-1.0, default 0.7)
- **RAS Window Length**: Controls repetition/pause patterns (0-15, default 7)
- **Max New Tokens**: Controls maximum audio length per chunk
- **Seed**: For reproducible results

## Best Practices

1. **Text Normalization**: The system automatically converts:
   - Chinese punctuation to English equivalents
   - Numbers should be written as words for best results
   - Parentheses are automatically removed

2. **Chunking Strategy**:
   - **Short text (<100 words)**: Use "none" chunking
   - **Multi-speaker dialog**: Use "speaker" chunking with System Placeholder Context
   - **Long single-speaker**: Use "word" chunking

3. **Context Strategy**:
   - **Enable System Placeholder Context** for best cross-chunk consistency
   - **Buffer Size 2-4**: Good balance of context and memory usage
   - **Buffer Size 5+**: Maximum context but higher memory usage

4. **Quality Control**: Enable quality checks for important projects, disable for faster iteration

## Debug Mode

When launched with `--debug`, the interface will print:
- Chunk processing details
- Quality check results and retry attempts
- Reference audio processing
- System placeholder context injection details
- Full system prompts
- Generation parameters

## Output Files

Generated files include suffixes to indicate features used:
- `_QC`: Quality control was enabled
- `_SYSPX`: System placeholder context was used
- `_chunked_[method]`: Chunking method used

"""
    return help_content

# ===== GRADIO INTERFACE =====

def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="HiggsAudio Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("HiggsAudio Generator")
        gr.Markdown("")
        
        if DEBUG_MODE:
            gr.Markdown("### 🐛 DEBUG MODE ENABLED - Check console for detailed output")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Main input section
                gr.Markdown("## 📝 Input")
                
                transcript = gr.Textbox(
                    label="Transcript",
                    placeholder="Enter your text here...\n\nFor multi-speaker: use [SPEAKER0], [SPEAKER1] tags\nFor single speaker: just enter your text",
                    lines=6,
                    max_lines=10
                )
                
                # Voice selection
                gr.Markdown("## 🎤 Voice Settings")
                
                use_reference_audio = gr.Checkbox(
                    label="Use Reference Audio/Voice",
                    value=False,
                    info="Enable to clone specific voices"
                )
                
                with gr.Row(visible=False) as voice_options:
                    with gr.Column():
                        gr.Markdown("### Upload Multiple Audio Files")
                        gr.Markdown("*Upload one file per speaker (SPEAKER0, SPEAKER1, etc.)*")
                        
                        # Create multiple file upload components
                        reference_audio_files = []
                        for i in range(5):  # Support up to 5 speakers
                            audio_input = gr.Audio(
                                label=f"Speaker {i} Reference Audio",
                                type="filepath",
                                visible=True if i < 2 else False  # Show first 2 by default
                            )
                            reference_audio_files.append(audio_input)
                        
                        show_more_speakers = gr.Button("+ Add More Speakers", size="sm")
                        
                    with gr.Column():
                        gr.Markdown("### Select Multiple Predefined Voices")
                        gr.Markdown("*Hold Ctrl/Cmd to select multiple*")
                        
                        predefined_voices = gr.Dropdown(
                            label="Predefined Voices",
                            choices=PREDEFINED_VOICES,
                            multiselect=True,
                            info="Choose from available voice profiles"
                        )
                
                # Voice descriptions section
                use_voice_descriptions = gr.Checkbox(
                    label="Use Voice Descriptions",
                    value=False,
                    info="Add custom voice characteristics for each speaker"
                )
                
                speaker_voice_descriptions = gr.Textbox(
                    label="Speaker Voice Descriptions",
                    placeholder="SPEAKER0: masculine; deep voice; british accent\nSPEAKER1: feminine; high pitch; excited tone",
                    lines=4,
                    visible=False,
                    info="One description per line. Use format: SPEAKERX: description"
                )
                
                # Scene settings
                gr.Markdown("## 🎬 Scene Settings")
                
                scene_prompt_key = gr.Dropdown(
                    label="Scene Preset",
                    choices=list(SCENE_PROMPTS.keys()),
                    value="Default (Quiet Room)",
                    info="Choose recording environment preset"
                )
                
                scene_prompt_text = gr.Textbox(
                    label="Scene Description (Editable)",
                    value=SCENE_PROMPTS["Default (Quiet Room)"],
                    lines=4,
                    info="Edit the scene description before generation"
                )
            
            with gr.Column(scale=1):
                # NEW: System Context Control
                gr.Markdown("## 🧠 Advanced Context Control (NEW)")
                
                use_system_placeholder_context = gr.Checkbox(
                    label="Use System Placeholder Context",
                    value=False,
                    info="🚀 EXPERIMENTAL: Inject previous chunks into system message for superior consistency"
                )
                
                with gr.Accordion("System Context Details", open=False):
                    gr.Markdown("""
                    **System Placeholder Context** is an advanced feature that uses HiggsAudio's native **AUDIO_PLACEHOLDER_TOKEN** support to inject recent audio chunks directly into the system message.
                    
                    **Benefits:**
                    - Superior cross-chunk voice consistency
                    - Better speaker identity preservation
                    - Advanced context awareness
                    
                    **Trade-offs:**
                    - Higher memory usage
                    - Slower generation
                    - More complex processing
                    
                    **Best for:** Multi-speaker conversations, long narratives, when voice consistency is critical
                    """)
                
                # Quality control settings
                gr.Markdown("## 🔍 Quality Control")
                
                enable_quality_checks = gr.Checkbox(
                    label="Enable Quality Checks",
                    value=True,
                    info="Automatically detect and regenerate problematic audio chunks"
                )
                
                with gr.Accordion("Quality Control Details", open=False):
                    gr.Markdown(f"""
                    **Current Settings:**
                    - Max retry attempts: {QUALITY_CHECK_PARAMS['MAX_REGENERATION_ATTEMPTS']}
                    - Duration tolerance: {QUALITY_CHECK_PARAMS['DURATION_TOLERANCE_FACTOR']}x expected
                    - Minimum duration: {QUALITY_CHECK_PARAMS['MIN_DURATION_FACTOR']}x expected
                    - Speech rate estimate: {QUALITY_CHECK_PARAMS['CHARS_PER_SECOND']} chars/second
                    - Repetition threshold: {QUALITY_CHECK_PARAMS['CORRELATION_THRESHOLD']}
                    
                    When enabled, chunks are automatically regenerated if they:
                    - Are too long/short compared to text length
                    - Are too quiet or silent
                    - Contain repetitive patterns
                    """)
                
                # Official chunking settings
                gr.Markdown("## 📋 Official Chunking Settings")
                
                chunk_method = gr.Dropdown(
                    label="Chunk Method",
                    choices=["none", "word", "speaker"],
                    value="none",
                    info="Official HiggsAudio chunking methods"
                )
                
                chunk_max_word_num = gr.Slider(
                    label="Chunk Max Words",
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=25,
                    info="For word chunking method"
                )
                
                chunk_max_num_turns = gr.Slider(
                    label="Chunk Max Turns",
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    info="For speaker chunking method"
                )
                
                generation_chunk_buffer_size = gr.Slider(
                    label="Generation Buffer Size",
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    info="Chunks to keep in memory/context"
                )
                
                ref_audio_in_system = gr.Checkbox(
                    label="Reference Audio in System Message",
                    value=False,
                    info="Include reference audio in system prompt"
                )
                
                # Advanced settings
                gr.Markdown("## ⚙️ Generation Settings")
                
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    info="Lower = more consistent, Higher = more creative"
                )
                
                top_p = gr.Slider(
                    label="Top-p (Nucleus Sampling)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05
                )
                
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1
                )
                
                max_new_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=128,
                    info="Controls maximum audio length"
                )
                
                seed = gr.Number(
                    label="Seed (0 = random)",
                    value=12345,
                    precision=0,
                    info="Use same seed for reproducible results"
                )
                
                # RAS settings
                ras_win_len = gr.Slider(
                    label="RAS Window Length",
                    minimum=0,
                    maximum=15,
                    value=7,
                    step=1,
                    info="Controls repetition/pause patterns. 0 = disabled"
                )
                
                ras_win_max_repeat = gr.Slider(
                    label="RAS Max Repeats", 
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    info="Maximum repetitions in RAS window"
                )
        
        # Control visibility and interactions
        def toggle_voice_options(use_ref):
            return gr.update(visible=use_ref)
        
        def toggle_voice_descriptions(use_desc):
            return gr.update(visible=use_desc)
        
        def update_scene_prompt(preset_key):
            return SCENE_PROMPTS.get(preset_key, "")
        
        def show_more_speaker_inputs():
            updates = []
            for i in range(5):
                updates.append(gr.update(visible=True))
            return updates
        
        def update_context_warning(use_context, chunk_method):
            if use_context and chunk_method != "none":
                return gr.update(value="🚀 System Placeholder Context ENABLED - Enhanced cross-chunk consistency active!")
            elif use_context:
                return gr.update(value="ℹ️ System Placeholder Context enabled but no chunking selected")
            else:
                return gr.update(value="")
        
        use_reference_audio.change(
            toggle_voice_options,
            inputs=[use_reference_audio],
            outputs=[voice_options]
        )
        
        use_voice_descriptions.change(
            toggle_voice_descriptions,
            inputs=[use_voice_descriptions],
            outputs=[speaker_voice_descriptions]
        )
        
        scene_prompt_key.change(
            update_scene_prompt,
            inputs=[scene_prompt_key],
            outputs=[scene_prompt_text]
        )
        
        show_more_speakers.click(
            show_more_speaker_inputs,
            outputs=reference_audio_files
        )
        
        # Generation section
        gr.Markdown("## 🎵 Generate")
        
        # Context status indicator
        context_status = gr.Markdown("")
        
        use_system_placeholder_context.change(
            update_context_warning,
            inputs=[use_system_placeholder_context, chunk_method],
            outputs=[context_status]
        )
        
        chunk_method.change(
            update_context_warning,
            inputs=[use_system_placeholder_context, chunk_method],
            outputs=[context_status]
        )
        
        with gr.Row():
            generate_btn = gr.Button(
                "🎵 Queue Generation (Advanced Features)", 
                variant="primary",
                size="lg"
            )
            
            help_btn = gr.Button(
                "❓ Help & Examples",
                variant="secondary"
            )
        
        # Queue status section
        with gr.Row():
            with gr.Column():
                queue_status = gr.HTML(value="<div style='text-align: center; color: #666;'>Queue: Empty</div>")
                
            with gr.Column():
                refresh_queue_btn = gr.Button("🔄 Refresh Queue", size="sm")
                check_results_btn = gr.Button("📥 Check Results", size="sm")
        
        # Output section
        gr.Markdown("## 📄 Output")
        
        with gr.Row():
            with gr.Column():
                output_audio = gr.Audio(
                    label="Latest Generated Audio",
                    type="filepath"
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column():
                gr.Markdown("### 📂 Recent Files")
                file_list = gr.HTML()
                refresh_files_btn = gr.Button("🔄 Refresh File List")
        
        # Help section (initially hidden)
        with gr.Row(visible=False) as help_section:
            with gr.Column():
                gr.Markdown("## 📖 Help & Documentation")
                help_content_display = gr.Markdown(create_help_content())
        
        # Track help visibility state
        help_visible = gr.State(False)
        
        def show_hide_help(visible_state):
            new_state = not visible_state
            return new_state, gr.update(visible=new_state)
        
        help_btn.click(
            show_hide_help,
            inputs=[help_visible],
            outputs=[help_visible, help_section]
        )
        
        # Queue management functions
        def get_queue_status_html():
            status = higgs_interface.generation_queue.get_queue_status()
            
            html = f"""
            <div style='padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px 0;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                    <span><strong>Queue:</strong> {status['queue_size']} waiting</span>
                    <span><strong>Completed:</strong> {status['completed']}</span>
                    <span><strong>Failed:</strong> {status['failed']}</span>
                </div>
                <div style='text-align: center; color: {'#28a745' if status['current_status'] != 'Idle' else '#6c757d'};'>
                    <strong>{status['current_status']}</strong>
                </div>
            </div>
            """
            return html
        
        def queue_generation(*args):
            try:
                # Package all the arguments - note the new parameter added
                job_data = {
                    'transcript': args[0],
                    'use_reference_audio': args[1],
                    'reference_audio_files': args[2:7],  # Extract the 5 audio file inputs
                    'predefined_voices': args[7],
                    'scene_prompt_text': args[8],
                    'speaker_voice_descriptions': args[9],
                    'use_voice_descriptions': args[10],
                    'temperature': args[11],
                    'top_p': args[12],
                    'top_k': args[13],
                    'max_new_tokens': args[14],
                    'seed': args[15],
                    'ras_win_len': args[16],
                    'ras_win_max_repeat': args[17],
                    'chunk_method': args[18],
                    'chunk_max_word_num': args[19],
                    'chunk_max_num_turns': args[20],
                    'generation_chunk_buffer_size': args[21],
                    'ref_audio_in_system': args[22],
                    'enable_quality_checks': args[23],
                    'use_system_placeholder_context': args[24]  # NEW parameter
                }
                
                job_id = higgs_interface.generation_queue.add_job(job_data)
                status = higgs_interface.generation_queue.get_queue_status()
                
                quality_note = " (Quality Control ON)" if job_data['enable_quality_checks'] else " (Quality Control OFF)"
                context_note = " (System Context ON)" if job_data['use_system_placeholder_context'] else ""
                
                return (
                    None,  # No immediate audio output
                    f"✅ Job {job_id} added to queue! Position: {status['queue_size']}{quality_note}{context_note}",
                    get_queue_status_html()
                )
            except Exception as e:
                higgs_interface.debug_print(f"Queue error: {str(e)}")
                return None, f"❌ Failed to queue job: {str(e)}", get_queue_status_html()
        
        def refresh_queue_status():
            return get_queue_status_html()
        
        def check_and_update_results():
            try:
                latest_job = higgs_interface.generation_queue.get_latest_completed_job()
                
                if latest_job and latest_job['status'] == 'completed':
                    return (
                        latest_job['audio_path'],
                        f"✅ {latest_job['message']} (Job: {latest_job['id']})",
                        get_queue_status_html(),
                        get_file_list(),
                        False  # Reset auto-refresh flag
                    )
                elif latest_job and latest_job['status'] == 'failed':
                    return (
                        None,
                        f"❌ Generation failed: {latest_job['error']} (Job: {latest_job['id']})",
                        get_queue_status_html(),
                        get_file_list(),
                        False  # Reset auto-refresh flag
                    )
                else:
                    return (
                        gr.update(),  # Keep current audio
                        gr.update(),  # Keep current status
                        get_queue_status_html(),
                        gr.update(),  # Keep current file list
                        False  # Reset auto-refresh flag
                    )
            except Exception as e:
                return (
                    gr.update(),
                    f"❌ Error checking results: {str(e)}",
                    get_queue_status_html(),
                    gr.update(),
                    False
                )
        
        def get_file_list():
            try:
                files = []
                if os.path.exists(OUTPUT_DIR):
                    for file in sorted(os.listdir(OUTPUT_DIR), reverse=True):
                        if file.endswith('.wav'):
                            file_path = os.path.join(OUTPUT_DIR, file)
                            size = os.path.getsize(file_path)
                            size_mb = size / (1024 * 1024)
                            
                            # Add quality control and context indicators
                            qc_indicator = " 🔍" if "_QC" in file else ""
                            context_indicator = " 🧠" if "_SYSPX" in file else ""
                            
                            files.append(f"<div style='margin: 5px 0; padding: 8px; background: #f0f0f0; border-radius: 4px;'>"
                                       f"<strong>{file}{qc_indicator}{context_indicator}</strong><br>"
                                       f"<small>Size: {size_mb:.2f} MB</small></div>")
                
                if files:
                    legend = "<div style='margin-bottom: 10px; font-size: 12px; color: #666;'>🔍 = Quality Control | 🧠 = System Context</div>"
                    return legend + "<div>" + "".join(files[:10]) + "</div>"  # Show last 10 files
                else:
                    return "<div style='text-align: center; color: #666;'>No generated files yet</div>"
            except:
                return "<div style='text-align: center; color: #666;'>Unable to load file list</div>"
        
        # Set up the result callback to trigger auto-refresh
        def setup_result_callback():
            def on_job_complete():
                try:
                    import threading
                    threading.Timer(0.5, lambda: check_results_btn.click()).start()
                except:
                    pass
            
            higgs_interface.generation_queue.result_callback = on_job_complete
        
        # Connect generate button to queue system - note the new parameter
        generate_btn.click(
            queue_generation,
            inputs=[
                transcript,
                use_reference_audio,
                *reference_audio_files,  # Unpack the 5 audio file inputs
                predefined_voices,
                scene_prompt_text,
                speaker_voice_descriptions,
                use_voice_descriptions,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                seed,
                ras_win_len,
                ras_win_max_repeat,
                chunk_method,
                chunk_max_word_num,
                chunk_max_num_turns,
                generation_chunk_buffer_size,
                ref_audio_in_system,
                enable_quality_checks,
                use_system_placeholder_context  # NEW parameter
            ],
            outputs=[output_audio, status_text, queue_status]
        )
        
        # Connect refresh buttons
        refresh_queue_btn.click(
            refresh_queue_status,
            outputs=[queue_status]
        )
        
        # Connect check results button
        check_results_btn.click(
            check_and_update_results,
            outputs=[output_audio, status_text, queue_status, file_list, gr.State()]
        )
        
        refresh_files_btn.click(
            get_file_list,
            outputs=[file_list]
        )
        
        # Load initial states
        interface.load(
            lambda: (get_file_list(), get_queue_status_html()),
            outputs=[file_list, queue_status]
        )
        
        # Set up result callback for auto-refresh
        interface.load(setup_result_callback)
    
    return interface

# ===== MAIN EXECUTION =====

def parse_arguments():
    parser = argparse.ArgumentParser(description="HiggsAudio Gradio Interface with Official Chunking, Quality Control, and System Placeholder Context")
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name/IP to bind to"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port to bind to"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed console output"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set global debug flag
    DEBUG_MODE = args.debug
    
    # Create and launch the interface
    interface = create_interface()
    
    print(f"🎵 HiggsAudio Generator Starting")
    print(f"📁 Audio files will be saved to: {OUTPUT_DIR}")
    print(f"🖥️ Device: {higgs_interface.device}")
    print(f"📋 Using HiggsAudio chunking methods")
    print(f"🔍 Quality control features enabled")
    print(f"🧠 System placeholder context support added")
    print(f"🔧 Single model load - optimized performance")
    
    if DEBUG_MODE:
        print(f"🐛 Debug mode enabled - detailed output will be shown in console")
        print(f"📊 Quality control parameters: {QUALITY_CHECK_PARAMS}")
        print(f"🎯 System context features: AUDIO_PLACEHOLDER_TOKEN injection")
    
    interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=DEBUG_MODE,
        show_error=True
    )