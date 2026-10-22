"""Single-node replica scheduling, checkpoints, and legacy JSON output.

Only workers import GPU libraries. Each replica gets disjoint visible GPUs
and may use vLLM tensor parallelism within that group.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

SCHEMA_VERSION = 1
MODES = {"emode2": "E-mode2", "emode3": "E-mode3"}
SEMANTIC_KEYS = (
    "backend", "model_path", "size", "adapter", "dtype", "quantization",
    "max_tokens", "temperature", "top_p", "top_k", "repetition_penalty",
    "seed", "trust_remote_code", "max_model_len", "engine_kwargs",
)


def atomic_json(path, value):
    """Never expose a partially written JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_name(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value.replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError(f"{label} must stay relative to its root: {value!r}")
    return path


def discover_tasks(text_root):
    """Match upstream's level/task[/subtask].json organization."""
    root = Path(text_root)
    if not root.is_dir():
        raise FileNotFoundError(f"FunBench text directory not found: {root}")
    tasks = []
    for level in sorted(root.iterdir()):
        if level.name.startswith(".") or not level.is_dir():
            continue
        for task in sorted(level.iterdir()):
            if task.name.startswith("."):
                continue
            if task.is_dir():
                tasks.extend(p for p in sorted(task.iterdir())
                             if p.is_file() and p.suffix == ".json" and not p.name.startswith("."))
            elif task.suffix == ".json":
                tasks.append(task)
    if not tasks:
        raise ValueError(f"No level/task[/subtask].json files found in {root}")
    return tasks


def build_requests(config):
    """Read prompts unchanged; both E-modes still receive their original image."""
    root = Path(config["text_root"]).resolve()
    image_root = Path(config["img_root"]).resolve()
    requests, missing, image_stamps = [], [], {}
    limit = config.get("limit")
    for task_file in discover_tasks(root):
        task = task_file.relative_to(root).as_posix()
        with task_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise ValueError(f"Expected an object with a 'data' object: {task_file}")
        for image_name, sample in data.items():
            relative_image = _relative_name(image_name, f"Image key in {task}")
            image_path = image_root.joinpath(*relative_image.parts)
            if not isinstance(sample, dict):
                raise ValueError(f"Invalid sample {image_name!r} in {task}")
            for mode in config["modes"]:
                field = MODES[mode]
                if field not in sample:
                    raise ValueError(f"Missing {field} for {image_name!r} in {task}")
                query = sample[field]
                if query is None and mode == "emode2":
                    continue
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(f"Expected nonempty {field} prompt for {image_name!r} in {task}")
                if str(image_path) not in image_stamps:
                    try:
                        if not image_path.is_file():
                            raise FileNotFoundError(image_path)
                        stat = image_path.stat()
                        image_stamps[str(image_path)] = [stat.st_size, stat.st_mtime_ns]
                    except FileNotFoundError:
                        image_stamps[str(image_path)] = None
                        missing.append(str(image_path))
                identity = json.dumps([task, mode, image_name], ensure_ascii=False)
                requests.append({
                    "id": hashlib.sha256(identity.encode()).hexdigest(),
                    "task": task, "mode": mode, "img_name": image_name,
                    "query": query, "img_path": str(image_path),
                    "image_stamp": image_stamps[str(image_path)],
                })
                if limit is not None and len(requests) >= limit:
                    break
            if limit is not None and len(requests) >= limit:
                break
        if limit is not None and len(requests) >= limit:
            break
    if missing:
        examples = "\n  ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} required images are missing under {image_root}. "
                                f"Use the preprocessed image root. Examples:\n  {examples}")
    if not requests:
        raise ValueError("The selected modes contain no prediction requests")
    return requests


def fingerprint(config, requests):
    semantics = {key: config.get(key) for key in SEMANTIC_KEYS}
    semantics['local_model_identity'] = local_model_identity(config['model_path'])
    serialized = json.dumps({"schema": SCHEMA_VERSION, "config": semantics,
                             "requests": requests}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def local_model_identity(model_path):
    """Detect local checkpoint updates without reading gigabytes of weights.

    Hash small model/processor configuration files; record file size and mtime
    for weights. Remote repository IDs must point to an immutable revision or
    be downloaded to an immutable local snapshot for reproducible resume.
    """
    root = Path(model_path)
    if not root.is_dir():
        return None
    identity = []
    suffixes = {'.json', '.jinja', '.jinja2', '.txt', '.model', '.tiktoken',
                '.safetensors', '.bin', '.pt', '.pth', '.py'}
    for current, directories, names in os.walk(root):
        directories[:] = sorted(name for name in directories if not name.startswith('.')
                                 and name != '__pycache__')
        for name in sorted(names):
            path = Path(current) / name
            if path.suffix not in suffixes or name.startswith('.'):
                continue
            stat = path.stat()
            entry = [path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns]
            if path.suffix not in {'.safetensors', '.bin', '.pt', '.pth'} and stat.st_size <= 16 * 1024**2:
                entry.append(hashlib.sha256(path.read_bytes()).hexdigest())
            identity.append(entry)
    return identity


def output_directory(config):
    name = config.get("run_name") or str(config["model_path"]).rstrip("/\\").replace("\\", "/").split("/")[-1]
    for value, label in [(name, "run_name"), (config["set_name"], "set_name")]:
        if len(_relative_name(value, label).parts) != 1:
            raise ValueError(f"{label} must be one directory name: {value!r}")
    return Path(config["out_root"]).resolve() / config["set_name"] / name


def _gpu_tokens(value):
    if value.strip() in ("", "-1"):
        return []
    tokens = [part.strip() for part in value.split(",")]
    if any(not part for part in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("GPU list must contain distinct, nonempty IDs")
    for token in tokens:
        if not token.isdigit() and not token.startswith(("GPU-", "MIG-")):
            raise ValueError(f"Invalid CUDA device identifier: {token!r}")
    return tokens


def gpu_groups(config, *, dry_run=False):
    """Resolve physical IDs/UUIDs in visible-device order without importing torch."""
    supplied, inherited = config.get("gpus"), os.environ.get("CUDA_VISIBLE_DEVICES")
    if supplied is not None:
        devices = _gpu_tokens(supplied)
        if inherited is not None and not set(devices).issubset(_gpu_tokens(inherited)):
            raise ValueError("--gpus must be a subset of the IDs/UUIDs in CUDA_VISIBLE_DEVICES; otherwise omit --gpus")
    elif inherited is not None:
        devices = _gpu_tokens(inherited)
    else:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                    capture_output=True, text=True, check=True, timeout=10)
            devices = _gpu_tokens(",".join(line.strip() for line in result.stdout.splitlines() if line.strip()))
        except (FileNotFoundError, subprocess.SubprocessError):
            devices = []
    if config["backend"] == "transformers":
        if not devices and not dry_run:
            raise RuntimeError("No visible NVIDIA GPUs. The original Transformers backend requires CUDA.")
        return [devices]
    dp, tp = config["data_parallel_size"], config["tensor_parallel_size"]
    needed = dp * tp
    if len(devices) < needed:
        if dry_run and not devices and supplied is None and inherited is None:
            return [None] * dp
        raise ValueError(f"DP={dp} × TP={tp} requires {needed} visible GPUs; found {len(devices)}: {devices}")
    return [devices[rank * tp:(rank + 1) * tp] for rank in range(dp)]


@contextlib.contextmanager
def output_lock(output):
    """OS releases the advisory lock even if a process crashes."""
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("The multi-GPU launcher requires Linux/POSIX. Use Linux/WSL2 for vLLM.") from exc
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".funbench.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another inference process is writing {output}") from exc
        try:
            handle.seek(0)
            handle.truncate()
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
            handle.flush()
            yield handle.fileno()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_journals(state, requests):
    """Only a torn final JSONL record may be discarded during crash recovery."""
    expected, answers = {request["id"] for request in requests}, {}
    for path in sorted(Path(state).glob("attempt-*/worker-*/answers.jsonl")):
        with path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                if not line.endswith(b"\n"):
                    break
                try:
                    record = json.loads(line)
                    request_id, answer = record["id"], record["text"]
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Corrupt checkpoint {path}:{number}") from exc
                if request_id not in expected or not isinstance(answer, str):
                    raise ValueError(f"Unexpected request or answer in {path}:{number}")
                if request_id in answers and answers[request_id] != answer:
                    raise ValueError(f"Conflicting answers for {request_id} in {path}")
                answers[request_id] = answer
    return answers


def merge_outputs(output, requests, answers):
    missing = [row["id"] for row in requests if row["id"] not in answers]
    if missing:
        raise RuntimeError(f"Cannot publish incomplete outputs: {len(missing)} requests missing")
    grouped = {}
    for row in requests:
        grouped.setdefault((row["mode"], row["task"]), {})[row["img_name"]] = answers[row["id"]]
    for (mode, task), data in grouped.items():
        atomic_json(Path(output) / mode / task, data)
    return len(grouped)


def _stop_workers(processes):
    # Include groups whose leader has already exited: vLLM grandchildren may live on.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def launch_workers(specs, groups, *, worker_module="funbench_inference.worker", lock_fd=None):
    """Each replica owns its process group, GPUs, and checkpoint journal."""
    processes, log_handles = [], []
    project_root = str(Path(__file__).resolve().parents[1])
    previous_handler = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    try:
        if hasattr(signal, "SIGTERM"):
            previous_handler = signal.signal(signal.SIGTERM, interrupted)
        for spec_path, devices in zip(specs, groups):
            env = os.environ.copy()
            for name in list(env):
                if name.startswith("VLLM_DP_") or name in (
                    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                    "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
                ):
                    env.pop(name)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            log_path = Path(spec_path).parent / "worker.log"
            handle = log_path.open("w", encoding="utf-8")
            log_handles.append(handle)
            print(f"Launching replica {len(processes)} on GPUs {devices}; log: {log_path}", flush=True)
            processes.append(subprocess.Popen(
                [sys.executable, "-m", worker_module, "--spec", str(spec_path)],
                env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                # Keep the lock held by live workers even if the parent is SIGKILLed.
                pass_fds=() if lock_fd is None else (lock_fd,),
            ))
        last_report = time.monotonic()
        while True:
            codes = [process.poll() for process in processes]
            for rank, code in enumerate(codes):
                if code not in (None, 0):
                    log_path = Path(specs[rank]).parent / "worker.log"
                    with log_path.open(encoding="utf-8", errors="replace") as handle:
                        tail = "".join(handle.readlines()[-30:])
                    raise RuntimeError(f"Replica {rank} failed (exit {code}). Log: {log_path}\n{tail}")
            if all(code == 0 for code in codes):
                break
            if time.monotonic() - last_report >= 10:
                done, total = 0, 0
                for path in specs:
                    progress_file = Path(path).parent / "progress.json"
                    if progress_file.exists():
                        progress = json.loads(progress_file.read_text(encoding="utf-8"))
                        done += progress["completed"]
                        total += progress["total"]
                print(f"Current attempt: {done}/{total} requests complete; "
                      f"{sum(code is None for code in codes)} replicas active", flush=True)
                last_report = time.monotonic()
            time.sleep(0.2)
    except BaseException:
        _stop_workers(processes)
        raise
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        for handle in log_handles:
            handle.close()


def run(config, *, worker_module="funbench_inference.worker"):
    requests = build_requests(config)
    output = output_directory(config)
    signature = fingerprint(config, requests)
    dry_run = config.get("dry_run", False)
    groups = gpu_groups(config, dry_run=dry_run)
    counts = {mode: sum(row["mode"] == mode for row in requests) for mode in config["modes"]}
    print(json.dumps({"output": str(output), "requests": len(requests), "per_mode": counts,
                      "backend": config["backend"], "gpu_groups": groups,
                      "tensor_parallel_size": config["tensor_parallel_size"],
                      "batch_size": config["batch_size"], "dry_run": dry_run}, indent=2), flush=True)
    if config.get("limit") is not None:
        print("WARNING: --limit produces a smoke-test subset, not a full benchmark run.", flush=True)
    if dry_run:
        print("Preflight passed. No models loaded and no output files written.", flush=True)
        return {"output": str(output), "requests": len(requests), "dry_run": True}
    with output_lock(output) as lock_fd:
        state = output / ".funbench_state"
        manifest_path = state / "manifest.json"
        if config.get("overwrite"):
            for path in (state, output / "emode2", output / "emode3"):
                if path.is_symlink():
                    raise ValueError(f"Refusing to overwrite a symlink: {path}")
            for path in (state, output / "emode2", output / "emode3"):
                if path.exists():
                    shutil.rmtree(path)
        if manifest_path.exists():
            if not config.get("resume"):
                raise FileExistsError(f"Run already exists: {output}. Use --resume, --overwrite, or another --run-name.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != signature:
                raise ValueError("Cannot resume: model, prompts, images, selected requests, or generation settings changed. "
                                 "Use another --run-name or --overwrite. DP/TP and batch size may change.")
        else:
            if any((output / mode).exists() for mode in MODES) or state.exists():
                raise FileExistsError("Existing outputs have no compatible checkpoint manifest. "
                                      "Use a new --run-name or --overwrite; legacy results are not imported implicitly.")
            state.mkdir(parents=True)
            manifest = {"schema": SCHEMA_VERSION, "fingerprint": signature, "config": config,
                        "request_count": len(requests), "status": "running"}
            atomic_json(manifest_path, manifest)
        answers = read_journals(state, requests)
        pending = [row for row in requests if row["id"] not in answers]
        print(f"Already checkpointed: {len(answers)}; pending: {len(pending)}", flush=True)
        if pending:
            manifest["status"] = "running"
            atomic_json(manifest_path, manifest)
            attempt = state / ("attempt-" + uuid.uuid4().hex)
            worker_count, specs = min(len(groups), len(pending)), []
            for rank in range(worker_count):
                worker_dir = attempt / f"worker-{rank}"
                spec = {"config": config, "requests": pending[rank::worker_count],
                        "worker_dir": str(worker_dir), "rank": rank}
                spec_path = worker_dir / "spec.json"
                atomic_json(spec_path, spec)
                atomic_json(worker_dir / "progress.json", {"completed": 0, "total": len(spec["requests"])})
                specs.append(spec_path)
            try:
                launch_workers(specs, groups[:worker_count], worker_module=worker_module, lock_fd=lock_fd)
            except BaseException:
                manifest["status"] = "interrupted"
                atomic_json(manifest_path, manifest)
                print("Completed batches are checkpointed. Rerun the same command with --resume.", flush=True)
                raise
            answers = read_journals(state, requests)
        file_count = merge_outputs(output, requests, answers)
        manifest.update({"status": "completed", "completed_requests": len(answers),
                         "output_files": file_count, "last_config": config})
        atomic_json(manifest_path, manifest)
        print(f"Done: {len(answers)} predictions in {file_count} JSON files under {output}", flush=True)
        return {"output": str(output), "requests": len(requests), "files": file_count, "dry_run": False}
