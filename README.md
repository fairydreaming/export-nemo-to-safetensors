This little script allows to convert NVIDIA Nemotron 4 340b model to HF safetensors format.
It's based on FailSpy script available here: https://github.com/FailSpy/export-nemo-to-safetensors
I fixed some bugs and modified it a bit.
Usage:
1. Install dependencies in pip or conda, required packages are: torch, tensorstore, zarr, numpy, safetensors, pyyaml, tqdm, sentencepiece, protobuf
2. Download the Nemotron 4 model
3. Run the `convert-nemo.py` script with two directories passed as arguments, the first one is the input directory, the second one is the output directory (it has to exist).

Example workflow:
```
pip install pip install torch tensorstore zarr numpy safetensors pyyaml tqdm sentencepiece protobuf
huggingface-cli download "nvidia/Nemotron-4-340B-Instruct"
mkdir ./Nemotron-4-safetensors
python3 convert-nemo.py ~/.cache/huggingface/hub/models--nvidia--Nemotron-4-340B-Instruct/snapshots/ac75bfbc2fb10d07fa90813707c18aebecdb9024/ ./Nemotron-4-safetensors
```

Note that at the time of writing Nemotron-4 is not supported in transformers library, so you can't use the resulting model with HF transformers.
Its sole purpose is to serve as an intermediate format for conversion to llama.cpp GGUF file.
