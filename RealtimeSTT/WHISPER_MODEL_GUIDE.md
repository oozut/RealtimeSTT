# Flexible WhisperModel Guide

This guide explains how to use the new flexible `WhisperModel` class that supports both internal faster_whisper models and external LLM transcription services via litellm.

## Overview

The new `WhisperModel` class is a drop-in replacement for `faster_whisper.WhisperModel` that provides:

1. **Internal Models**: Use local faster_whisper models (default behavior)
2. **External Models**: Use external LLM transcription services via litellm
3. **Seamless Switching**: Switch between models without changing your code
4. **Environment Configuration**: Configure via environment variables or parameters

## Usage Examples

### 1. Internal Model (Default)
```python
from RealtimeSTT.whisper_model import WhisperModel

# Use local faster_whisper model (default)
model = WhisperModel("base", device="cuda")
segments, info = model.transcribe("audio.wav")
```

### 2. External Model via OpenAI
```python
import os
from RealtimeSTT.whisper_model import WhisperModel

# Configure to use OpenAI's Whisper API
os.environ["WHISPER_MODEL_TYPE"] = "external"
os.environ["WHISPER_LITELLM_MODEL"] = "openai/whisper-1"
os.environ["WHISPER_LITELLM_API_KEY"] = "your-openai-api-key"

model = WhisperModel("whisper-1")
segments, info = model.transcribe("audio.wav")
```

### 3. External Model via Anthropic
```python
import os
from RealtimeSTT.whisper_model import WhisperModel

# Configure to use Anthropic's transcription
os.environ["WHISPER_MODEL_TYPE"] = "external"
os.environ["WHISPER_LITELLM_MODEL"] = "anthropic/claude-3-sonnet-20240229"
os.environ["WHISPER_LITELLM_API_KEY"] = "your-anthropic-api-key"

model = WhisperModel("claude-3-sonnet-20240229")
segments, info = model.transcribe("audio.wav")
```

### 4. Programmatic Configuration
```python
from RealtimeSTT.whisper_model import WhisperModel

# Configure via parameters
model = WhisperModel(
    "whisper-1",
    model_type="external",
    litellm_model="openai/whisper-1",
    litellm_api_key="your-api-key"
)
segments, info = model.transcribe("audio.wav")
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `WHISPER_MODEL_TYPE` | Model type: "internal" or "external" | "external" |
| `WHISPER_LITELLM_MODEL` | Model identifier for litellm | "openai/whisper-1" |
| `WHISPER_LITELLM_API_KEY` | API key for external service | "sk-..." |
| `WHISPER_LITELLM_BASE_URL` | Custom base URL (optional) | "https://api.openai.com/v1" |

## Supported External Models

The following models are supported via litellm:

- **OpenAI**: `openai/whisper-1`
- **Anthropic**: `anthropic/claude-3-sonnet-20240229`
- **Google**: `google/speech-to-text`
- **Azure**: `azure/speech-to-text`
- **Local**: Any OpenAI-compatible API endpoint

## Configuration Examples

### Docker Environment
```dockerfile
ENV WHISPER_MODEL_TYPE=external
ENV WHISPER_LITELLM_MODEL=openai/whisper-1
ENV WHISPER_LITELLM_API_KEY=sk-your-key-here
```

### Python Script
```python
import os
from RealtimeSTT import AudioToTextRecorder

# Configure external model
os.environ["WHISPER_MODEL_TYPE"] = "external"
os.environ["WHISPER_LITELLM_MODEL"] = "openai/whisper-1"
os.environ["WHISPER_LITELLM_API_KEY"] = "your-api-key"

# Use with RealtimeSTT
recorder = AudioToTextRecorder(model="whisper-1")
text = recorder.text()
```

### Configuration File
```python
# config.py
import os

# Internal model configuration
os.environ["WHISPER_MODEL_TYPE"] = "internal"
recorder = AudioToTextRecorder(model="base")

# External model configuration
os.environ["WHISPER_MODEL_TYPE"] = "external"
os.environ["WHISPER_LITELLM_MODEL"] = "openai/whisper-1"
os.environ["WHISPER_LITELLM_API_KEY"] = "sk-..."
recorder = AudioToTextRecorder(model="whisper-1")
```

## Installation

### For Internal Models (Default)
```bash
pip install RealtimeSTT
```

### For External Models
```bash
pip install RealtimeSTT[external]
# or
pip install litellm
```

## API Compatibility

The new `WhisperModel` maintains 100% API compatibility with `faster_whisper.WhisperModel`:

```python
# All these methods work identically
model = WhisperModel("base")
segments, info = model.transcribe(
    audio,
    language="en",
    beam_size=5,
    temperature=0.0,
    initial_prompt="Transcribe this audio"
)

# Access segment properties
for segment in segments:
    print(segment.text)  # Works the same
    print(segment.start)  # Works the same
    print(segment.end)  # Works the same

# Access info properties
print(info.language)  # Works the same
print(info.language_probability)  # Works the same
```

## Error Handling

### Missing litellm
```python
try:
    from RealtimeSTT.whisper_model import WhisperModel
    model = WhisperModel("whisper-1", model_type="external")
except ImportError as e:
    print("Install litellm: pip install litellm")
```

### Invalid Configuration
```python
try:
    model = WhisperModel("whisper-1", model_type="external")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Performance Considerations

| Model Type | Latency | Cost | Privacy |
|------------|---------|------|---------|
| Internal | Low | Free | High |
| External | Medium-High | Paid | Medium |

## Migration Guide

### From faster_whisper to flexible WhisperModel

1. **Replace imports**:
   ```python
   # Old
   from faster_whisper import WhisperModel
   
   # New
   from RealtimeSTT.whisper_model import WhisperModel
   ```

2. **No code changes needed** for internal models
3. **Add environment variables** for external models

### Backward Compatibility
The new `WhisperModel` is fully backward compatible. Existing code using `faster_whisper.WhisperModel` will continue to work without any changes.

## Troubleshooting

### Common Issues

1. **litellm not found**
   ```bash
   pip install litellm
   ```

2. **API key not provided**
   ```python
   os.environ["WHISPER_LITELLM_API_KEY"] = "your-key"
   ```

3. **Model not supported**
   ```python
   # Check supported models
   import litellm
   print(litellm.models)
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

model = WhisperModel("whisper-1", model_type="external")
```

## Advanced Usage

### Custom Base URL
```python
model = WhisperModel(
    "whisper-1",
    model_type="external",
    litellm_model="openai/whisper-1",
    litellm_api_key="your-key",
    litellm_base_url="https://your-custom-endpoint.com/v1"
)
```

### Runtime Switching
```python
import os
from RealtimeSTT.whisper_model import WhisperModel

# Start with internal model
os.environ["WHISPER_MODEL_TYPE"] = "internal"
model1 = WhisperModel("base")

# Switch to external model
os.environ["WHISPER_MODEL_TYPE"] = "external"
os.environ["WHISPER_LITELLM_MODEL"] = "openai/whisper-1"
model2 = WhisperModel("whisper-1")
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the litellm documentation for external model support
3. Open an issue on the RealtimeSTT GitHub repository
