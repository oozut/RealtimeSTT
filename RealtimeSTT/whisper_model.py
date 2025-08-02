"""
Flexible WhisperModel implementation supporting both internal faster_whisper
and external LLM transcription via litellm.

This module provides a drop-in replacement for faster_whisper.WhisperModel that
can use either:
1. Local faster_whisper models (internal)
2. External LLM transcription services via litellm (external)

Usage:
    # Internal model (default behavior)
    model = WhisperModel("base", device="cuda")
    
    # External model via litellm
    import os
    os.environ["WHISPER_MODEL_TYPE"] = "external"
    os.environ["WHISPER_LITELLM_MODEL"] = "openai/whisper-1"
    model = WhisperModel("whisper-1")  # Uses OpenAI's Whisper API
"""

import os
import logging
import tempfile
import numpy as np
from typing import Union, List, Tuple, Optional, Iterator, Any
import io

# Try to import litellm, but don't fail if not available
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Import faster_whisper for internal models
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


class Segment:
    """Represents a transcribed segment, compatible with faster_whisper format."""
    
    def __init__(self, text: str, start: float = 0.0, end: float = 0.0, 
                 words: Optional[List] = None):
        self.text = text
        self.start = start
        self.end = end
        self.words = words or []
    
    def __repr__(self):
        return f"Segment(text='{self.text}', start={self.start}, end={self.end})"


class TranscriptionInfo:
    """Represents transcription information, compatible with faster_whisper format."""
    
    def __init__(self, language: str = "en", language_probability: float = 1.0,
                 duration: float = 0.0, transcription_options: Optional[dict] = None):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration
        self.transcription_options = transcription_options or {}
    
    def __repr__(self):
        return f"TranscriptionInfo(language='{self.language}', probability={self.language_probability})"


class WhisperModel:
    """
    Flexible WhisperModel that supports both internal faster_whisper models
    and external LLM transcription services via litellm.
    
    This class provides a drop-in replacement for faster_whisper.WhisperModel
    with the same interface but extended functionality.
    """
    
    def __init__(
        self,
        model_size_or_path: str = "base",
        device: str = "auto",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs
    ):
        """
        Initialize the WhisperModel.
        
        Args:
            model_size_or_path: Model identifier or path
            device: Device to use ('auto', 'cpu', 'cuda')
            device_index: GPU device index
            compute_type: Compute precision type
            cpu_threads: Number of CPU threads
            num_workers: Number of workers
            download_root: Root directory for model downloads
            local_files_only: Use only local files
            **kwargs: Additional arguments for external models
        """
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root
        self.local_files_only = local_files_only
        
        # Determine model type from environment or kwargs
        self.model_type = os.getenv("WHISPER_MODEL_TYPE", "internal").lower()
        self.litellm_model = os.getenv("WHISPER_LITELLM_MODEL", model_size_or_path)
        self.litellm_api_key = os.getenv("WHISPER_LITELLM_API_KEY")
        self.litellm_base_url = os.getenv("WHISPER_LITELLM_BASE_URL")
        
        # Override with kwargs if provided
        if "model_type" in kwargs:
            self.model_type = kwargs.pop("model_type")
        if "litellm_model" in kwargs:
            self.litellm_model = kwargs.pop("litellm_model")
        if "litellm_api_key" in kwargs:
            self.litellm_api_key = kwargs.pop("litellm_api_key")
        if "litellm_base_url" in kwargs:
            self.litellm_base_url = kwargs.pop("litellm_base_url")
        
        self.kwargs = kwargs
        
        # Initialize the appropriate model
        if self.model_type == "external":
            self._init_external_model()
        else:
            self._init_internal_model()
    
    def _init_internal_model(self):
        """Initialize internal faster_whisper model."""
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is not available. Install with: "
                "pip install faster-whisper"
            )
        
        self.internal_model = FasterWhisperModel(
            model_size_or_path=self.model_size_or_path,
            device=self.device,
            device_index=self.device_index,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
            download_root=self.download_root,
            local_files_only=self.local_files_only,
            **self.kwargs
        )
        self.external_model = None
        
    def _init_external_model(self):
        """Initialize external model via litellm."""
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not available. Install with: "
                "pip install litellm"
            )
        
        self.internal_model = None
        self.external_model = {
            "model": self.litellm_model,
            "api_key": self.litellm_api_key,
            "base_url": self.litellm_base_url,
            **self.kwargs
        }
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        temperature: Union[float, List[float], Tuple[float, ...]] = 0.0,
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        max_initial_timestamp: float = 1.0,
        **kwargs
    ) -> Tuple[Iterator[Segment], TranscriptionInfo]:
        """
        Transcribe audio using the configured model.
        
        Args:
            audio: Audio data (file path, numpy array, or bytes)
            language: Language code (e.g., 'en', 'es')
            task: Task type ('transcribe' or 'translate')
            beam_size: Beam search size
            best_of: Number of candidates when sampling
            patience: Beam search patience
            length_penalty: Length penalty
            temperature: Temperature for sampling
            compression_ratio_threshold: Compression ratio threshold
            logprob_threshold: Log probability threshold
            no_speech_threshold: No speech threshold
            condition_on_previous_text: Condition on previous text
            initial_prompt: Initial prompt
            word_timestamps: Return word timestamps
            prepend_punctuations: Prepend punctuations
            append_punctuations: Append punctuations
            max_initial_timestamp: Maximum initial timestamp
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (segments iterator, transcription info)
        """
        if self.model_type == "external":
            return self._transcribe_external(
                audio,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations,
                max_initial_timestamp=max_initial_timestamp,
                **kwargs
            )
        else:
            return self._transcribe_internal(
                audio,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations,
                max_initial_timestamp=max_initial_timestamp,
                **kwargs
            )
    
    def _transcribe_internal(self, audio, **kwargs) -> Tuple[Iterator[Segment], TranscriptionInfo]:
        """Transcribe using internal faster_whisper model."""
        segments, info = self.internal_model.transcribe(audio, **kwargs)
        
        # Convert faster_whisper segments to our Segment format
        def segment_generator():
            for segment in segments:
                yield Segment(
                    text=segment.text,
                    start=getattr(segment, 'start', 0.0),
                    end=getattr(segment, 'end', 0.0),
                    words=getattr(segment, 'words', [])
                )
        
        # Convert info to our TranscriptionInfo format
        transcription_info = TranscriptionInfo(
            language=getattr(info, 'language', 'en'),
            language_probability=getattr(info, 'language_probability', 1.0),
            duration=getattr(info, 'duration', 0.0)
        )
        
        return segment_generator(), transcription_info
    
    def _transcribe_external(self, audio, **kwargs) -> Tuple[Iterator[Segment], TranscriptionInfo]:
        """Transcribe using external LLM via litellm."""
        # Convert audio to appropriate format
        audio_data = self._prepare_audio_for_external(audio)
        
        # Prepare transcription request
        messages = [
            {
                "role": "user",
                "content": "Transcribe this audio to text."
            }
        ]
        
        # Prepare parameters for litellm
        transcription_params = {
            "model": self.external_model["model"],
            "audio": audio_data,
            "response_format": "text"
        }
        
        # Add language if specified
        language = kwargs.get("language")
        if language:
            transcription_params["language"] = language
        
        # Add API key and base URL if provided
        if self.external_model.get("api_key"):
            transcription_params["api_key"] = self.external_model["api_key"]
        if self.external_model.get("base_url"):
            transcription_params["base_url"] = self.external_model["base_url"]
        
        try:
            # Use litellm for transcription
            response = litellm.transcription(**transcription_params)
            
            # Extract text from response
            if hasattr(response, 'text'):
                text = response.text
            elif isinstance(response, dict) and 'text' in response:
                text = response['text']
            else:
                text = str(response)
            
            # Create segments and info
            segments = [Segment(text=text.strip())]
            info = TranscriptionInfo(
                language=language or "en",
                language_probability=1.0
            )
            
            return iter(segments), info
            
        except Exception as e:
            logging.error(f"Error in external transcription: {e}")
            # Fallback to empty transcription
            segments = [Segment(text="")]
            info = TranscriptionInfo(language=language or "en", language_probability=0.0)
            return iter(segments), info
    
    def _prepare_audio_for_external(self, audio) -> bytes:
        """Prepare audio data for external API."""
        if isinstance(audio, str):
            # File path - read the file
            with open(audio, 'rb') as f:
                return f.read()
        elif isinstance(audio, np.ndarray):
            # Convert numpy array to bytes
            # Assume 16-bit PCM, 16kHz, mono
            if audio.dtype != np.int16:
                # Normalize and convert to int16
                audio = (audio * 32767).astype(np.int16)
            return audio.tobytes()
        elif isinstance(audio, bytes):
            return audio
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
    
    def __repr__(self):
        return f"WhisperModel(model_type='{self.model_type}', model='{self.model_size_or_path}')"


# Alias for backward compatibility
WhisperModel = WhisperModel
