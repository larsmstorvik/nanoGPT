import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXP_DIR = BASE_DIR / "experiment_results_20260128_154716"
RESULTS_DIR = BASE_DIR / "results"


@dataclass
class RunInfo:
    sweep: str
    param_name: str
    param_value: str
    training_log: Path
    samples_file: Path
    info_file: Path


def parse_training_log(log_path: Path) -> Optional[Dict[str, np.ndarray]]:
    iters = []
    train_losses = []
    eval_steps = []
    eval_train_losses = []
    eval_val_losses = []
    vocab_size = None

    if not log_path.exists():
        print(f"[WARN] No log found at {log_path}")
        return None

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"iter\s+(\d+):\s+loss\s+([0-9]*\.?[0-9]+)", line)
            if m:
                iters.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                continue

            m = re.search(
                r"step\s+(\d+):\s+train loss\s+([0-9]*\.?[0-9]+),\s+val loss\s+([0-9]*\.?[0-9]+)",
                line,
            )
            if m:
                eval_steps.append(int(m.group(1)))
                eval_train_losses.append(float(m.group(2)))
                eval_val_losses.append(float(m.group(3)))
                continue

            m = re.search(r"found vocab_size = (\d+)", line)
            if m:
                vocab_size = int(m.group(1))

    if not iters and not eval_steps:
        print(f"[WARN] No lines matched in {log_path}")
        return None

    return {
        "iter": np.array(iters),
        "train_loss": np.array(train_losses),
        "eval_step": np.array(eval_steps),
        "eval_train_loss": np.array(eval_train_losses),
        "eval_val_loss": np.array(eval_val_losses),
        "vocab_size": np.array([vocab_size]) if vocab_size is not None else np.array([]),
    }


def parse_info_file(info_path: Path) -> Tuple[str, str]:
    param_name = "param"
    param_value = "?"
    if not info_path.exists():
        return param_name, param_value

    with info_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Parameter varied:"):
                # Example: Parameter varied: n_layer = 2
                parts = line.split(":", 1)[1].strip().split("=")
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_value = parts[1].strip()
                break
            if line.startswith("Parameter value:"):
                # Fallback: Parameter value: 2
                param_value = line.split(":", 1)[1].strip()
    return param_name, param_value


def discover_runs(exp_dir: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    if not exp_dir.exists():
        print(f"[WARN] Experiment directory not found: {exp_dir}")
        return runs

    for sweep_dir in sorted(exp_dir.iterdir()):
        if not sweep_dir.is_dir():
            continue
        sweep = sweep_dir.name
        for training_log in sorted(sweep_dir.glob("*_training_log.txt")):
            stem = training_log.stem.replace("_training_log", "")
            info_file = sweep_dir / f"{stem}_info.txt"
            samples_file = sweep_dir / f"{stem}_samples.txt"
            param_name, param_value = parse_info_file(info_file)
            runs.append(
                RunInfo(
                    sweep=sweep,
                    param_name=param_name,
                    param_value=param_value,
                    training_log=training_log,
                    samples_file=samples_file,
                    info_file=info_file,
                )
            )
    return runs


def plot_loss_curves(runs: List[RunInfo], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run in runs:
        data = parse_training_log(run.training_log)
        if not data:
            continue
        label = f"{run.param_name}={run.param_value}"
        if len(data["iter"]) > 0:
            plt.plot(data["iter"], data["train_loss"], label=f"train {label}")
        if len(data["eval_step"]) > 0:
            plt.plot(
                data["eval_step"],
                data["eval_val_loss"],
                marker="o",
                linestyle="--",
                label=f"val {label}",
            )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def final_loss_bar(runs: List[RunInfo], title: str, output_path: Path) -> None:
    labels = []
    values = []
    for run in runs:
        data = parse_training_log(run.training_log)
        if not data or len(data["train_loss"]) == 0:
            continue
        labels.append(f"{run.param_value}")
        values.append(float(data["train_loss"][-1]))

    if not values:
        print(f"[WARN] No final losses available for {title}")
        return

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#4C78A8")
    plt.xlabel("Parameter value")
    plt.ylabel("Final train loss")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def tokenize_simple(text: str) -> List[str]:
    return text.strip().split()


def distinct_n(texts: List[str], n: int = 1) -> float:
    all_ngrams = []
    total = 0
    for t in texts:
        tok = tokenize_simple(t)
        if len(tok) < n:
            continue
        ngrams = [tuple(tok[i : i + n]) for i in range(len(tok) - n + 1)]
        all_ngrams += ngrams
        total += len(ngrams)
    return len(set(all_ngrams)) / total if total > 0 else 0.0


def type_token_ratio(texts: List[str]) -> float:
    toks = []
    for t in texts:
        toks += tokenize_simple(t)
    return len(set(toks)) / len(toks) if toks else 0.0


def load_samples(samples_path: Path) -> List[str]:
    if not samples_path.exists():
        return []
    raw = samples_path.read_text(encoding="utf-8")

    start_idx = raw.find("Loading meta")
    if start_idx != -1:
        raw = raw[start_idx:]
        # Skip header up to the first blank line after "Loading meta"
        parts = raw.split("\n\n", 1)
        if len(parts) == 2:
            raw = parts[1]

    raw = raw.replace("---------------", "")
    chunks = [c.strip() for c in raw.split("<|endoftext|>")]
    return [c for c in chunks if c]


def summarize_diversity(runs: List[RunInfo]) -> List[str]:
    lines = [f"{'run':<25}{'#samples':<10}{'distinct-1':<12}{'distinct-2':<12}{'TTR':<8}"]
    for run in runs:
        samples = load_samples(run.samples_file)
        if not samples:
            continue
        d1 = distinct_n(samples, 1)
        d2 = distinct_n(samples, 2)
        ttr = type_token_ratio(samples)
        label = f"{run.param_name}={run.param_value}"
        lines.append(f"{label:<25}{len(samples):<10}{d1:<12.3f}{d2:<12.3f}{ttr:<8.3f}")
    return lines


def approx_params(n_layer: int, n_embd: int, vocab_size: int, block_size: int = 256) -> int:
    d = n_embd
    block = 12 * (d**2) * n_layer
    embed = vocab_size * d + block_size * d
    return block + embed


def parse_model_dims(training_log: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    n_layer = None
    n_head = None
    n_embd = None
    with training_log.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("n_layer"):
                n_layer = int(line.split("=")[1].strip())
            elif line.strip().startswith("n_head"):
                n_head = int(line.split("=")[1].strip())
            elif line.strip().startswith("n_embd"):
                n_embd = int(line.split("=")[1].strip())
            if n_layer is not None and n_head is not None and n_embd is not None:
                break
    return n_layer, n_head, n_embd


def summarize_params(runs: List[RunInfo]) -> List[str]:
    lines = [f"{'run':<25}{'n_layer':<8}{'n_head':<8}{'n_embd':<8}{'params(M)':<10}"]
    for run in runs:
        data = parse_training_log(run.training_log)
        if not data:
            continue
        n_layer, n_head, n_embd = parse_model_dims(run.training_log)
        vocab_size = int(data["vocab_size"][0]) if len(data["vocab_size"]) > 0 else 228
        if n_layer is None or n_embd is None:
            continue
        params = approx_params(n_layer, n_embd, vocab_size) / 1e6
        label = f"{run.param_name}={run.param_value}"
        lines.append(
            f"{label:<25}{str(n_layer):<8}{str(n_head):<8}{str(n_embd):<8}{params:<10.2f}"
        )
    return lines


def show_samples(run: RunInfo, k: int = 2, max_chars: int = 400) -> List[str]:
    lines = []
    samples = load_samples(run.samples_file)
    for i, s in enumerate(samples[:k], 1):
        snippet = s[:max_chars].replace("\n", " ")
        lines.append(f"--- {run.param_name}={run.param_value} sample {i} ---")
        lines.append(snippet)
        lines.append("...")
        lines.append("")
    return lines


def group_runs_by_sweep(runs: List[RunInfo]) -> Dict[str, List[RunInfo]]:
    grouped: Dict[str, List[RunInfo]] = {}
    for run in runs:
        grouped.setdefault(run.sweep, []).append(run)
    for sweep in grouped:
        grouped[sweep] = sorted(grouped[sweep], key=lambda r: float(r.param_value))
    return grouped


def write_lines(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(exp_dir: Path = DEFAULT_EXP_DIR, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(exp_dir)
    if not runs:
        return

    grouped = group_runs_by_sweep(runs)

    for sweep, sweep_runs in grouped.items():
        loss_path = results_dir / f"{sweep}_loss_curves.png"
        bar_path = results_dir / f"{sweep}_final_train_loss.png"
        plot_loss_curves(sweep_runs, title=f"Loss Curves - {sweep}", output_path=loss_path)
        final_loss_bar(sweep_runs, title=f"Final Train Loss - {sweep}", output_path=bar_path)

        lines = []
        lines.append(f"Sweep: {sweep}")
        lines.append("")
        lines.append("Diversity metrics:")
        lines.extend(summarize_diversity(sweep_runs))
        lines.append("")
        lines.append("Approx parameter sizes:")
        lines.extend(summarize_params(sweep_runs))
        if sweep_runs:
            lines.append("")
            lines.append("Samples:")
            lines.extend(show_samples(sweep_runs[0]))

        write_lines(results_dir / f"{sweep}_summary.txt", lines)


if __name__ == "__main__":
    main()
