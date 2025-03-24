
# FoleyMultiGen

Foley is a crucial part of videomaking, but is also very labor intensive. That's why FoleyMultiGen aims at generating audio which correspon to a video, with an optional text and audio guiding.

Nicoas GAUTTIER
Ivain GUITTARD

### Prepare Environment
After cloning the repo, use the following command to install dependencies:
```bash
# install virtual environment
python3 venv -m venv venvfoley
source venvfoley/bin/activate

# install depedencies
pip install requirements/txt
```

### Download Checkpoints
First make sure your code editor is connected with hugging face
The checkpoints will be downloaded automatically by running `inference.py`.


Put checkpoints as follows:
```
└── checkpoints
    ├── semantic
    │   ├── semantic_adapter.bin
    ├── vocoder
    │   ├── vocoder.pt
    │   ├── config.json
    ├── temporal_adapter.ckpt
    │   │
    └── timestamp_detector.pth.tar
```

## Inference
### Video To Audio Generation
```bash
python inference.py --input==path_to_input_folder --save_dir==path_to_save_directory
```

You can use the following commands to add text and audio prompts

### Commandline Usage Parameters
```console
options:
  --prompt PROMPT       prompt for audio generation
  --nprompt NPROMPT     negative prompt for audio generation
  --seed SEED           ramdom seed
  --temporal_scale TEMPORAL_SCALE
                        temporal align scale
  --semantic_scale SEMANTIC_SCALE
                        visual content scale
  --input INPUT         input video folder path
  --ckpt CKPT           checkpoints folder path
  --save_dir SAVE_DIR   generation result save path
  --device DEVICE
  --audio_prompt_path          path_to_audio_prompt
```