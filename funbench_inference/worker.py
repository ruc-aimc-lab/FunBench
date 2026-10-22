"""Private worker entry point. CUDA visibility is set before this module loads."""

import argparse
import json
import os
from pathlib import Path

from .runner import atomic_json


def execute(spec_path, predictor_factory=None):
    with Path(spec_path).open(encoding="utf-8") as handle:
        spec = json.load(handle)
    config, requests = spec["config"], spec["requests"]
    worker_dir = Path(spec["worker_dir"])
    if predictor_factory is None:
        from .backends import create_predictor
        predictor_factory = create_predictor
    print(f"Replica {spec['rank']}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}; "
          f"{len(requests)} requests", flush=True)
    predictor = predictor_factory(config)
    completed = 0
    # Each attempt owns a fresh journal. No two processes ever write this file.
    with (worker_dir / "answers.jsonl").open("x", encoding="utf-8") as handle:
        for start in range(0, len(requests), config["batch_size"]):
            batch = requests[start:start + config["batch_size"]]
            outputs = predictor.generate_batch(batch)
            if len(outputs) != len(batch) or any(not isinstance(text, str) for text in outputs):
                raise RuntimeError("Predictor returned a wrong number or type of answers")
            for request, answer in zip(batch, outputs):
                if not answer.strip():
                    raise RuntimeError(f"Empty model output for {request['task']} / {request['mode']} / "
                                       f"{request['img_name']}; no placeholder answer is saved")
            for request, answer in zip(batch, outputs):
                handle.write(json.dumps({"id": request["id"], "text": answer}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed += len(batch)
            atomic_json(worker_dir / "progress.json", {"completed": completed, "total": len(requests)})
            print(f"Replica {spec['rank']}: {completed}/{len(requests)} requests", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    execute(parser.parse_args().spec)


if __name__ == "__main__":
    main()
