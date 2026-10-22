"""Lazy-loaded image/text inference backends for FunBench.

The original Qwen2.5-VL ``Predictor(model_path, size=224)`` API is retained.
GPU libraries are deliberately imported only when a backend is constructed:
the launcher must set CUDA_VISIBLE_DEVICES before constructing a predictor.

vLLM's public offline API and input format are documented at:
https://docs.vllm.ai/en/stable/features/multimodal_inputs/
https://docs.vllm.ai/en/stable/serving/offline_inference/
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from typing import Any


# These options are owned by the launcher / named CLI arguments. Letting an
# arbitrary JSON override change GPU topology would break worker isolation.
PROTECTED_ENGINE_KWARGS = frozenset({
    "model", "tokenizer", "tensor_parallel_size", "pipeline_parallel_size",
    "data_parallel_size", "data_parallel_size_local", "data_parallel_rank",
    "data_parallel_start_rank", "distributed_executor_backend", "dtype",
    "trust_remote_code", "gpu_memory_utilization", "max_model_len",
    "max_num_seqs", "quantization", "seed", "enforce_eager",
    "generation_config", "limit_mm_per_prompt",
    "revision", "tokenizer_revision", "code_revision",
    "override_generation_config", "device_ids", "nnodes", "node_rank",
})


def validate_engine_kwargs(engine_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate extra engine settings without importing inference libraries."""
    if engine_kwargs is not None and not isinstance(engine_kwargs, Mapping):
        raise TypeError("engine_kwargs must be a JSON object / mapping.")
    extra = dict(engine_kwargs or {})
    if any(not isinstance(key, str) for key in extra):
        raise TypeError("engine_kwargs keys must be strings.")
    blocked = PROTECTED_ENGINE_KWARGS.intersection(extra) | {
        key for key in extra
        if key.startswith(("data_parallel_", "pipeline_parallel_", "tensor_parallel_"))
    }
    if blocked:
        raise ValueError(
            "engine_kwargs cannot override launcher/CLI-managed arguments: "
            + ", ".join(sorted(blocked))
        )
    return extra


def _import_dependency(name: str, install_hint: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import {name!r}. {install_hint} Original import error: {exc}"
        ) from exc


def _validate_request(request: Mapping[str, Any]) -> tuple[str, str]:
    query = request.get("query")
    img_path = request.get("img_path")
    if not isinstance(query, str):
        raise TypeError("Every inference request must contain a string 'query'.")
    if not isinstance(img_path, (str, os.PathLike)):
        raise TypeError("Every inference request must contain a path 'img_path'.")
    return query, os.fspath(img_path)


class _ImagePromptBuilder:
    """Build exactly one image + one question using the model's chat template."""

    def __init__(self, model_path: str, size: int = 224,
                 adapter: str = "auto", trust_remote_code: bool = False):
        if size < 0:
            raise ValueError("Image size must be >= 0; 0 keeps native aspect ratio.")
        if adapter not in {"auto", "qwen", "generic"}:
            raise ValueError("adapter must be auto, qwen, or generic.")
        transformers = _import_dependency(
            "transformers", "Install the inference backend requirements."
        )
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.size = size
        # Processor identity also handles locally renamed / fine-tuned models.
        # Qwen3 has a different patch size and intentionally uses the generic
        # PIL route, letting its own processor perform the final resizing.
        processor_name = type(self.processor).__name__.lower()
        self.adapter = (
            "qwen" if "qwen2" in processor_name and "vl" in processor_name
            else "generic"
        ) if adapter == "auto" else adapter
        self._process_vision_info = None
        if self.adapter == "qwen":
            qwen_utils = _import_dependency(
                "qwen_vl_utils", "Install qwen-vl-utils for the Qwen adapter."
            )
            self._process_vision_info = qwen_utils.process_vision_info

    def prepare(self, query: str, img_path: str) -> tuple[str, list[Any]]:
        image_item: dict[str, Any] = {"type": "image", "image": img_path}
        if self.size > 0 and self.adapter == "qwen":
            image_item.update(resized_height=self.size, resized_width=self.size)
        messages = [{
            "role": "user",
            "content": [image_item, {"type": "text", "text": query}],
        }]
        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "The selected model processor cannot format the image/text chat. "
                "Use a checkpoint with a multimodal chat template or adapt "
                "_ImagePromptBuilder.prepare() for that architecture."
            ) from exc
        if not isinstance(prompt, str):
            raise TypeError("The processor chat template must return a string.")

        if self.adapter == "qwen":
            images, videos = self._process_vision_info(messages)
            if videos is not None:
                raise ValueError("FunBench prediction expects image inputs only.")
            if images is None or len(images) != 1:
                raise ValueError("The Qwen adapter must produce exactly one image.")
            return prompt, images

        pil_image = _import_dependency("PIL.Image", "Install Pillow.")
        # Detach the RGB image before closing the file, so a batch never leaves
        # hundreds of file descriptors open while the model is running.
        with pil_image.open(img_path) as source:
            image = source.convert("RGB")
        if self.size > 0:
            image = image.resize(
                (self.size, self.size), resample=pil_image.Resampling.BICUBIC
            )
        return prompt, [image]


class VLLMPredictor:
    """One vLLM engine, optionally tensor-parallel across its visible GPUs.

    ``generate_batch`` performs a single batched call. Multiple engines (data
    parallelism) are launched by the outer runner on disjoint device groups.
    """

    def __init__(
        self, model_path: str, size: int = 224, *,
        tensor_parallel_size: int = 1, dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None, max_num_seqs: int = 64,
        max_tokens: int = 256, temperature: float | None = 0.0,
        top_p: float | None = 1.0, top_k: int | None = -1,
        repetition_penalty: float | None = 1.0, seed: int = 0,
        trust_remote_code: bool = False, enforce_eager: bool = False,
        quantization: str | None = None, adapter: str = "auto",
        engine_kwargs: Mapping[str, Any] | None = None,
    ):
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1.")
        if max_tokens < 1 or max_num_seqs < 1:
            raise ValueError("max_tokens and max_num_seqs must be >= 1.")
        if not 0.0 < gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1].")
        extra = validate_engine_kwargs(engine_kwargs)
        vllm = _import_dependency(
            "vllm", "Install a compatible vLLM build using requirements-vllm.txt."
        )
        self._builder = _ImagePromptBuilder(
            model_path, size, adapter, trust_remote_code
        )
        self.processor = self._builder.processor
        self.size = size
        self.adapter = self._builder.adapter
        modality_limits = {"image": 1}
        processor_name = type(self.processor).__name__.lower()
        if (("qwen" in processor_name and "vl" in processor_name)
                or getattr(self.processor, "video_processor", None) is not None):
            # Image-only evaluation must not reserve memory to profile unused
            # video inputs. See vLLM's "Conserving Memory" guide.
            modality_limits["video"] = 0
        # Explicit sampling prevents a checkpoint's generation_config.json from
        # silently imposing a different sampling policy / max_new_tokens cap.
        self.sampling_params = vllm.SamplingParams(
            n=1, max_tokens=max_tokens,
            temperature=0.0 if temperature is None else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=-1 if top_k is None else top_k,
            repetition_penalty=(
                1.0 if repetition_penalty is None else repetition_penalty
            ),
            seed=seed,
        )
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
            "seed": seed,
            "limit_mm_per_prompt": modality_limits,
            "generation_config": "vllm",
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if quantization is not None:
            llm_kwargs["quantization"] = quantization
        if tensor_parallel_size > 1:
            # Every replica is local to the CUDA_VISIBLE_DEVICES group assigned
            # by the launcher; no Ray auto-detection across replicas is wanted.
            llm_kwargs["distributed_executor_backend"] = "mp"
        llm_kwargs.update(extra)
        self.model = vllm.LLM(**llm_kwargs)

    def generate(self, query: str, img_path: str) -> str:
        return self.generate_batch([{"query": query, "img_path": img_path}])[0]

    def generate_batch(self, requests: Sequence[Mapping[str, Any]]) -> list[str]:
        if not requests:
            return []
        prompts = []
        for request in requests:
            query, img_path = _validate_request(request)
            prompt, images = self._builder.prepare(query, img_path)
            prompts.append({"prompt": prompt, "multi_modal_data": {"image": images}})
        results = self.model.generate(
            prompts, sampling_params=self.sampling_params, use_tqdm=False
        )
        # LLM.generate guarantees input order, even when request lengths differ.
        if len(results) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(results)} results for {len(requests)} requests."
            )
        texts = []
        for index, result in enumerate(results):
            if len(result.outputs) != 1 or not isinstance(result.outputs[0].text, str):
                raise RuntimeError(f"Invalid vLLM completion for batch item {index}.")
            texts.append(result.outputs[0].text)
        return texts


class Predictor:
    """Transformers fallback with the original ``generate`` interface.

    A direct ``Predictor(model_path)`` preserves the model's generation config.
    Explicit sampling arguments override it; temperature=0 requests greedy
    decoding. This compatibility backend intentionally processes one question
    at a time, matching the original FunBench behavior.
    """

    def __init__(
        self, model_path: str, size: int = 224, *, dtype: str = "auto",
        max_tokens: int = 256, temperature: float | None = None,
        top_p: float | None = None, top_k: int | None = None,
        repetition_penalty: float | None = None, seed: int | None = None,
        trust_remote_code: bool = False, adapter: str = "auto",
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1.")
        self._torch = _import_dependency(
            "torch", "Install the Transformers inference requirements."
        )
        transformers = _import_dependency(
            "transformers", "Install the Transformers inference requirements."
        )
        self._builder = _ImagePromptBuilder(
            model_path, size, adapter, trust_remote_code
        )
        self.processor = self._builder.processor
        self.size = size
        self.adapter = self._builder.adapter
        processor_name = type(self.processor).__name__.lower()
        if "qwen2_5" in processor_name or "qwen2.5" in model_path.lower():
            model_class = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        elif "qwen2" in processor_name and "vl" in processor_name:
            model_class = getattr(transformers, "Qwen2VLForConditionalGeneration", None)
        else:
            model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            raise RuntimeError(
                "This Transformers installation has no compatible image/text "
                "generation class. Upgrade Transformers or use the vLLM backend."
            )
        torch_dtype = dtype
        if dtype in {"half", "float16"}:
            torch_dtype = self._torch.float16
        elif dtype in {"bfloat16"}:
            torch_dtype = self._torch.bfloat16
        elif dtype in {"float", "float32"}:
            torch_dtype = self._torch.float32
        elif dtype != "auto":
            raise ValueError("Unsupported dtype for the Transformers backend.")
        if seed is not None:
            transformers.set_seed(seed)
        self.model = model_class.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.generation_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
        if temperature is not None:
            if temperature < 0:
                raise ValueError("temperature must be >= 0.")
            self.generation_kwargs["do_sample"] = temperature > 0
            if temperature > 0:
                self.generation_kwargs["temperature"] = temperature
        if top_p is not None:
            self.generation_kwargs["top_p"] = top_p
        if top_k is not None:
            # vLLM uses -1 for an unrestricted top-k; Transformers uses 0.
            self.generation_kwargs["top_k"] = max(0, top_k)
        if repetition_penalty is not None:
            self.generation_kwargs["repetition_penalty"] = repetition_penalty

    def generate(self, query: str, img_path: str) -> str:
        _validate_request({"query": query, "img_path": img_path})
        prompt, images = self._builder.prepare(query, os.fspath(img_path))
        processor_kwargs = {
            "text": [prompt], "images": images,
            "padding": True, "return_tensors": "pt",
        }
        if self.adapter == "qwen":
            processor_kwargs["videos"] = None
        inputs = self.processor(**processor_kwargs)
        # The first model device also works when device_map='auto' offloads some
        # layers to CPU. Avoid assuming that the default CUDA index is correct.
        inputs = inputs.to(self.model.device)
        with self._torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.generation_kwargs)
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if len(outputs) != 1 or not isinstance(outputs[0], str):
            raise RuntimeError("Transformers returned an invalid completion.")
        return outputs[0]

    def generate_batch(self, requests: Sequence[Mapping[str, Any]]) -> list[str]:
        answers = []
        for request in requests:
            query, img_path = _validate_request(request)
            answers.append(self.generate(query, img_path))
        return answers


def create_predictor(config: Mapping[str, Any]):
    """Construct a predictor from runner config, ignoring orchestration keys."""
    backend = config.get("backend", "vllm")
    common = {
        "model_path", "size", "dtype", "max_tokens", "temperature", "top_p",
        "top_k", "repetition_penalty", "seed", "trust_remote_code", "adapter",
    }
    if backend == "vllm":
        keys = common | {
            "tensor_parallel_size", "gpu_memory_utilization", "max_model_len",
            "max_num_seqs", "enforce_eager", "quantization", "engine_kwargs",
        }
        return VLLMPredictor(**{key: config[key] for key in keys if key in config})
    if backend in {"transformers", "hf"}:
        return Predictor(**{key: config[key] for key in common if key in config})
    raise ValueError(f"Unknown inference backend: {backend!r}")
