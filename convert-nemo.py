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

    added_tokens=dict()
    for i, piece in enumerate(sentencepiece_model.pieces):
        if piece.type == sentencepiece_model.SentencePiece.USER_DEFINED:
            added_tokens[piece.piece] = i

    print("Writing added_tokens.json")
    (output_dir/'added_tokens.json').write_text(json.dumps(added_tokens, indent=2))

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

    model_conversion_plan = list()

    file_num = 1
    model_conversion_plan.append((file_num, (('special', 'embedding.word_embeddings.weight'),)))
    file_num += 1

    for layer in range(layer_count):
        file_conversion_plan = []
        for key in model_map.keys():
            file_conversion_plan.append(('layer', key, layer))
        model_conversion_plan.append((file_num, tuple(file_conversion_plan)))
        file_num += 1

    file_conversion_plan = []
    for tensor_name in ['final_layernorm.weight', 'final_layernorm.bias', 'output_layer.weight']:
        file_conversion_plan.append(('special', tensor_name))
    model_conversion_plan.append((file_num, tuple(file_conversion_plan)))
    file_num += 1

    for (file_num, file_conversion_plan) in model_conversion_plan:
        sharded_state_dict = dict()
        fname = f"model-{file_num:05}-of-{len(model_conversion_plan):05}.safetensors"

        for tensor_conversion_plan in file_conversion_plan:
            tensor_type = tensor_conversion_plan[0]
            key = tensor_conversion_plan[1]
            k = layer_mappings[key]
            if tensor_type == "special":
                arr = special_layers[key]
            elif tensor_type == "layer":
                layer_num = tensor_conversion_plan[2]
                arr = model_map[key]
                arr = arr[layer_num,:]
                k = k.replace("{lnum}",str(layer_num))

            print(f"converting {key} of shape {arr.shape} to {k} ")
            sharded_state_dict[k] = convert_to_torch(arr)
            index[k] = fname

        save_file(sharded_state_dict,output_dir/fname)

        # cleanup to save RAM
        del sharded_state_dict
        gc.collect()

        print("saved",fname)

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
