#!/usr/bin/env python3

import torch
import tensorstore # needed for bfloat16 on zarr
import zarr
import numpy as np
from pathlib import Path
from safetensors.torch import save_file
import gc
from tqdm import tqdm
from collections import OrderedDict
import json
import argparse
import os
import yaml
import shutil

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from sentencepiece import SentencePieceProcessor
from sentencepiece import sentencepiece_model_pb2 as model

layer_mappings = {
        'layers.mlp.linear_fc1.layer_norm_bias': 'model.layers.{lnum}.mlp.linear_fc1.layer_norm.bias',
        'layers.mlp.linear_fc1.layer_norm_weight': 'model.layers.{lnum}.mlp.linear_fc1.layer_norm.weight',
        'layers.mlp.linear_fc1.weight': 'model.layers.{lnum}.mlp.linear_fc1.weight',
        'layers.mlp.linear_fc2.weight': 'model.layers.{lnum}.mlp.linear_fc2.weight',
        'layers.self_attention.linear_qkv.weight': 'model.layers.{lnum}.self_attention.linear_qkv.weight',
        'layers.self_attention.linear_proj.weight': 'model.layers.{lnum}.self_attention.linear_proj.weight',
        'layers.self_attention.linear_qkv.layer_norm_bias': 'model.layers.{lnum}.self_attention.linear_qkv.layer_norm.bias',
        'layers.self_attention.linear_qkv.layer_norm_weight': 'model.layers.{lnum}.self_attention.linear_qkv.layer_norm.weight',
        'embedding.word_embeddings.weight': 'embedding.word_embeddings.weight',
        'final_layernorm.weight': 'final_layernorm.weight',
        'final_layernorm.bias': 'final_layernorm.bias',
        'output_layer.weight': 'output_layer.weight'
}

def convert_to_torch(tensor):
    if "bfloat16" in tensor.dtype.name:
        # bfloat16 isn't properly supported by numpy, so gotta convert to a different format then back
        tensor = torch.from_numpy(tensor.view(np.int16)).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(tensor)
    return tensor

activation_mapping = {
    "squared-relu": "relu2",
}

def convert_nemo_config(model_dir: Path, output_dir: Path):
    print("Reading model_config.yaml")
    with (model_dir/'model_config.yaml').open() as f:
        model_config = yaml.safe_load(f);

    tokenizer_model_filename = model_config["tokenizer"]["tokenizer_model"].removeprefix("nemo:")
    tokenizer_path = model_dir/tokenizer_model_filename

    print("Reading tokenizer model from " + str(tokenizer_path))
    sentencepiece_model = model.ModelProto()
    sentencepiece_model.ParseFromString(open(tokenizer_path, "rb").read())

    sp = SentencePieceProcessor()
    sp.LoadFromFile(str(tokenizer_path))

    print("Writing tokenizer model")
    shutil.copy2(tokenizer_path, output_dir/"tokenizer.model")

    print("Creating config.json")
    hf_config = {
        "architectures": ["NemotronForCausalLM"],
        "attention_bias": model_config["bias"],
        "attention_dropout": model_config["attention_dropout"],
        "unk_token_id": sentencepiece_model.trainer_spec.unk_id,
        "bos_token_id": sentencepiece_model.trainer_spec.bos_id,
        "eos_token_id": sentencepiece_model.trainer_spec.eos_id,
        "pad_token_id": sentencepiece_model.trainer_spec.pad_id,
        "hidden_act": activation_mapping[model_config["activation"]],
        "hidden_size": model_config["hidden_size"],
        "initializer_range": 0.0063,
        "intermediate_size": model_config["ffn_hidden_size"],
        "max_position_embeddings": model_config["max_position_embeddings"],
        "model_type": "nemotron",
        "num_attention_heads": model_config["num_attention_heads"],
        "num_hidden_layers": model_config["num_layers"],
        "num_key_value_heads": model_config["num_query_groups"],
        "pretraining_tp": 1,
        "layer_norm_eps": model_config["layernorm_epsilon"],
        "rope_scaling": None,
        "partial_rotary_factor": model_config["rotary_percentage"], 
        "rope_theta": float(model_config["rotary_base"]) if "rotary_base" in model_config else 10000.0,
        "tie_word_embeddings": model_config["share_embeddings_and_output_weights"],
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0.dev0",
        "use_cache": True,
        "vocab_size": sp.get_piece_size()
    }
    print("Writing config.json")
    (output_dir/'config.json').write_text(json.dumps(hf_config, indent=2))

def convert_nemo_model(model_dir: Path, output_dir: Path):
    model_map = {}
    layer_count = 0
    special_layers = {}

    for subdir in (model_dir/"model_weights").iterdir():
        if not subdir.is_dir() or not (subdir / '.zarray').exists():
            continue
        sharded_state_dict = {}
        key = subdir.name

        arr = zarr.convenience.open(subdir,'r')

        key = key.split('.')
        while key[0] in ('model','decoder'):
            key.pop(0)

        multi_layered = key[0] == 'layers'
        key = '.'.join(key)

        if not multi_layered:
            arr = np.expand_dims(arr,0)
            special_layers[key] = arr
        else:
            if layer_count < arr.shape[0]:
                layer_count = arr.shape[0]
            model_map[key] = arr

    print("Exporting", layer_count, "layers")

    # have the index ordered mostly for readability's sake
    index = OrderedDict()

    # we store the output layer at the end in its own file, and keep it at top of index
    index['output_layer.weight'] = f"model-{layer_count+1:05}-of-{layer_count+1:05}.safetensors"
    output_layer = convert_to_torch(special_layers['output_layer.weight'])
    fname = f"model-{layer_count+1:05}-of-{layer_count+1:05}.safetensors"
    save_file({'output_layer.weight':output_layer},output_dir/fname)

    # now that we have instances to each, let's store things by order of layers for better loading
    for layer in range(layer_count):
        # hacky way of positioning standalone layers:
        if layer == 0:
            model_map['embedding.word_embeddings.weight'] = special_layers['embedding.word_embeddings.weight']
        elif layer == layer_count-1:
            model_map['final_layernorm.weight'] = special_layers['final_layernorm.weight']
            model_map['final_layernorm.bias'] = special_layers['final_layernorm.bias']

        sharded_state_dict = dict()
        fname = f"model-{layer+1:05}-of-{layer_count+1:05}.safetensors"

        for key,arr in tqdm(model_map.items()):
            lnum = layer
            print(f"{key}: {arr.shape}")
            if arr.shape[0] <= layer:
                lnum = 0
            k = layer_mappings[key].replace("{lnum}",str(layer))
            sharded_state_dict[k] = convert_to_torch(arr[lnum,:])
            index[k] = fname

        save_file(sharded_state_dict,output_dir/fname)

        # cleanup to save RAM
        del sharded_state_dict
        gc.collect()

        print("saved",fname)
        if layer == 0:
            del model_map['embedding.word_embeddings.weight']

    print("done, writing index")
    safetensor_index = OrderedDict()
    safetensor_index['metadata'] = OrderedDict()
    safetensor_index['metadata']['total_size'] = 0
    safetensor_index['weight_map'] = index
    (output_dir/'model.safetensors.index.json').write_text(json.dumps(safetensor_index))

def dir_path(string):
    if os.path.isdir(string):
        return Path(string)
    else:
        raise NotADirectoryError(string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help = "Directory containing the Nemo model files", type=dir_path)
    parser.add_argument("output_dir", help = "Output directory", type=dir_path)
    args = parser.parse_args()
    convert_nemo_config(args.model_dir, args.output_dir)
    convert_nemo_model(args.model_dir, args.output_dir)
