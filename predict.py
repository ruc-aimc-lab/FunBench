"""FunBench image-text inference using vLLM or the original Transformers backend.

CLI defaults to vLLM. Existing imports of Predictor and five-argument calls to
main(img_root, text_root, set_name, model_path, out_root) retain Transformers.
"""

import argparse
import json
import os
from pathlib import Path
import sys

from funbench_inference.backends import Predictor, VLLMPredictor, validate_engine_kwargs
from funbench_inference.runner import run


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img-root', default='datasets_preprocessed', help='Preprocessed image directory')
    parser.add_argument('--text-root', default='FunBench', help='Downloaded FunBench prompt directory')
    parser.add_argument('--model-path', default='Qwen/Qwen2.5-VL-7B-Instruct', help='HF model ID or local checkpoint')
    parser.add_argument('--out-root', default='answers')
    parser.add_argument('--set-name', default='test')
    parser.add_argument('--run-name', help='Output directory name; defaults to model basename')
    parser.add_argument('--backend', choices=['vllm', 'transformers'], default='vllm')
    parser.add_argument('--adapter', choices=['auto', 'qwen', 'generic'], default='auto')
    parser.add_argument('--gpus', help='Comma-separated physical GPU IDs/UUIDs; otherwise use CUDA_VISIBLE_DEVICES')
    parser.add_argument('--data-parallel-size', '--dp', type=int, default=1, help='Independent replicas on disjoint GPUs')
    parser.add_argument('--tensor-parallel-size', '--tp', type=int, default=1, help='GPUs per vLLM replica')
    parser.add_argument('--batch-size', type=int, default=32, help='Requests submitted per replica per checkpoint batch')
    parser.add_argument('--size', type=int, default=224, help='Square image size; 0 keeps source aspect ratio for model preprocessing')
    parser.add_argument('--dtype', choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'], default='auto')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--max-model-len', type=int, default=None, help='vLLM total context length; defaults to model config')
    parser.add_argument('--max-num-seqs', type=int, default=32, help='Concurrent sequences in each vLLM engine')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=None, help='vLLM defaults to greedy 0; Transformers uses model config')
    parser.add_argument('--top-p', type=float, default=None, help='vLLM default 1; Transformers keeps model config')
    parser.add_argument('--top-k', type=int, default=None, help='vLLM default -1; Transformers keeps model config')
    parser.add_argument('--repetition-penalty', type=float, default=None, help='vLLM default 1; Transformers keeps model config')
    parser.add_argument('--seed', type=int, default=None, help='vLLM default 0; Transformers keeps its existing RNG state')
    parser.add_argument('--quantization', help='vLLM quantization method for a compatible checkpoint')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--enforce-eager', action='store_true', help='Disable vLLM CUDA graphs')
    parser.add_argument('--engine-kwargs', default='{}', help='Additional vLLM LLM kwargs as JSON (topology/model options protected)')
    parser.add_argument('--modes', nargs='+', choices=['emode2', 'emode3'], default=['emode2', 'emode3'])
    parser.add_argument('--limit', type=int, help='Maximum prediction requests for smoke testing only')
    parser.add_argument('--dry-run', action='store_true', help='Check paths, prompts and GPU allocation without loading a model')
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument('--resume', action='store_true', help='Continue a matching checkpointed run')
    existing.add_argument('--overwrite', action='store_true', help="Replace this run's checkpoints and both E-mode output directories")
    return parser


def validate_config(config):
    for key in ('img_root', 'text_root', 'model_path', 'out_root'):
        config[key] = os.fspath(config[key])
    for name in ('data_parallel_size', 'tensor_parallel_size', 'batch_size', 'max_num_seqs', 'max_tokens'):
        if config[name] < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if config['size'] < 0:
        raise ValueError('--size must be nonnegative')
    if config.get('limit') is not None and config['limit'] < 1:
        raise ValueError('--limit must be positive')
    if config['max_model_len'] is not None and config['max_model_len'] <= config['max_tokens']:
        raise ValueError('--max-model-len must exceed --max-tokens to leave room for image and prompt tokens')
    if not 0 < config['gpu_memory_utilization'] <= 1:
        raise ValueError('--gpu-memory-utilization must be in (0, 1]')
    if config['backend'] == 'vllm':
        for key, default in {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1,
                             'repetition_penalty': 1.0, 'seed': 0}.items():
            if config[key] is None:
                config[key] = default
    if config['temperature'] is not None and config['temperature'] < 0:
        raise ValueError('--temperature must be nonnegative')
    if config['top_p'] is not None and not 0 < config['top_p'] <= 1:
        raise ValueError('--top-p must be in (0, 1]')
    if config['top_k'] is not None and (config['top_k'] == 0 or config['top_k'] < -1):
        raise ValueError('--top-k must be -1 or positive')
    if config['repetition_penalty'] is not None and config['repetition_penalty'] <= 0:
        raise ValueError('--repetition-penalty must be positive')
    if config['seed'] is not None and (config['seed'] < 0 or config['seed'] >= 2**32):
        raise ValueError('--seed must be in [0, 2**32)')
    if config.get('resume') and config.get('overwrite'):
        raise ValueError('--resume and --overwrite are mutually exclusive')
    if isinstance(config['engine_kwargs'], str):
        config['engine_kwargs'] = json.loads(config['engine_kwargs'])
    config['engine_kwargs'] = validate_engine_kwargs(config['engine_kwargs'])
    config['modes'] = list(dict.fromkeys(config['modes']))
    if not config['modes'] or any(mode not in ('emode2', 'emode3') for mode in config['modes']):
        raise ValueError('modes must contain emode2 and/or emode3')
    if config['backend'] == 'transformers':
        if config['data_parallel_size'] != 1 or config['tensor_parallel_size'] != 1:
            raise ValueError("Explicit DP/TP requires --backend vllm; Transformers retains device_map='auto'")
        if config['engine_kwargs'] or config['quantization']:
            raise ValueError('--engine-kwargs and --quantization are vLLM options')
    if Path(config['model_path']).exists():
        config['model_path'] = str(Path(config['model_path']).resolve())
    return config


def main(img_root, text_root, set_name, model_path, out_root, **kwargs):
    """Original five-argument Python entry; opt into vLLM with a keyword."""
    config = vars(build_parser().parse_args([]))
    config.update(img_root=img_root, text_root=text_root, set_name=set_name,
                  model_path=model_path, out_root=out_root, backend='transformers')
    unknown = set(kwargs) - set(config)
    if unknown:
        raise TypeError(f"Unknown inference options: {', '.join(sorted(unknown))}")
    config.update(kwargs)
    return run(validate_config(config))


def cli():
    parser = build_parser()
    try:
        run(validate_config(vars(parser.parse_args())))
    except KeyboardInterrupt:
        print('Interrupted. Checkpoints are retained; use --resume to continue.', file=sys.stderr)
        return 130
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(cli())
