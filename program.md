# AutoHarness Stress Test

Replicating the AutoHarness paper (Google DeepMind, arXiv 2603.03329) in a new domain: numerical code generation. The research question: does a small model (Qwen-2.5-Coder 1.5B) + auto-synthesized constraint harness beat a large model (Claude) without one?

Full spec: `docs/superpowers/specs/2026-03-14-autoharness-stress-test-design.md`

## Setup

Work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoharness/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoharness/<tag>` from current main.
3. **Read the spec**: `docs/superpowers/specs/2026-03-14-autoharness-stress-test-design.md` — this is the full design.
4. **Verify prerequisites**:
   - Ollama is running: `ollama list` should show `qwen2.5-coder:1.5b`. If not: `ollama pull qwen2.5-coder:1.5b`.
   - `ANTHROPIC_API_KEY` is set in env.
   - `uv` is installed.
5. **Bootstrap the project** (if not already done):
   - Create `pyproject.toml` with dependencies: `anthropic`, `ollama`, `numpy`, `scipy`.
   - `uv sync`
   - Create the directory structure: `problems/`, `harnesses/`, `solutions/`.
   - Create the 30 problem specs as JSON files in `problems/` (see Problem Benchmark below).
   - Create `generate.py` — wrapper for Claude API and Ollama calls.
   - Create `evaluate.py` — loads a harness and runs it against a solution.
   - Create `sweep.py` — orchestrates a single experiment (one problem + one tightness level + both conditions).
   - Initialize `results.tsv` with the header row.
6. **Confirm and go**: confirm setup looks good, then start the experiment loop.

## Problem Benchmark

Create 30 JSON files in `problems/`, 10 per difficulty level.

Each file follows this format:
```json
{
  "id": "matrix_inverse",
  "difficulty": "medium",
  "description": "Implement a function that computes the inverse of an NxN matrix. Must handle singular matrices by raising ValueError. Numerical precision within 1e-6 of numpy.linalg.inv.",
  "function_signature": "def matrix_inverse(m: list[list[float]]) -> list[list[float]]",
  "constraints": [
    "must handle singular matrices by raising ValueError",
    "numerical precision within 1e-6"
  ]
}
```

**Easy (10):** dot_product, matrix_transpose, polynomial_eval, gcd, euclidean_distance, vector_normalize, running_mean, power_function, modular_exp, linear_interpolation

**Medium (10):** matrix_inverse, fft, simpsons_integration, gaussian_elimination, matrix_determinant, cholesky, newtons_method, cubic_spline, convolution, lu_decomposition

**Hard (10):** svd, eigenvalue_decomposition, rk4_ode_solver, sparse_matmul, conjugate_gradient, qr_decomposition, lanczos, numerical_laplace, multivariate_newton, pseudoinverse

## Architecture

### generate.py

Two functions:

```python
def call_claude(prompt: str, system: str = "") -> str:
    """Call Claude API. Returns generated code as string."""

def call_qwen(prompt: str, system: str = "") -> str:
    """Call Ollama Qwen 2.5 Coder 1.5B. Set num_ctx=8192. Returns generated code as string."""
```

### evaluate.py

```python
def run_harness(harness_code: str, solution_code: str) -> dict:
    """
    Executes harness against solution in a subprocess.
    Returns {"passed": bool, "score": float, "violations": list[str]}
    """
```

Run in a subprocess with a 30-second timeout. Catch all exceptions. Never execute untrusted code in the main process.

### sweep.py

```python
def run_experiment(problem_id: str, tightness: str) -> dict:
    """
    Runs one experiment:
    1. Load problem spec from problems/<problem_id>.json
    2. Generate harness: call Claude with the problem spec + tightness instructions
    3. Save harness to harnesses/<problem_id>_<tightness>.py
    4. Run Treatment (Qwen + harness):
       - Qwen generates code
       - Run harness, get violations
       - If violations, feed them back to Qwen, retry (up to 5 rounds)
       - Record: passed, score, rounds_used, total_tokens
    5. Run Baseline (Claude, no harness feedback):
       - Claude generates code
       - Run same harness to evaluate
       - If fail, retry with "tests failed, try again" (up to 5 rounds)
       - Record: passed, score, rounds_used, total_tokens
    6. Return results dict for both conditions
    """
```

## Harness Synthesis Prompts

When asking Claude to generate a harness, use this structure:

**Minimal tightness:**
> Generate a Python test harness for this function. Only check: correct return type, correct output shape/dimensions, no crashes on basic inputs. Return format: `(passed: bool, score: float, violations: list[str])`.

**Standard tightness:**
> Generate a Python test harness for this function. Check: correct return type and shape, mathematical properties (e.g. A @ inv(A) ≈ I), 5+ edge cases (empty, zero, singular, large values, negative). Use numpy/scipy as reference. Return format: `(passed: bool, score: float, violations: list[str])`.

**Strict tightness:**
> Generate a Python test harness for this function. Check: correct return type and shape, all mathematical properties, 10+ edge cases, 20 random inputs compared against numpy/scipy reference within tolerance 1e-6. Return format: `(passed: bool, score: float, violations: list[str])`.

## Logging Results

Log every experiment to `results.tsv` (tab-separated). Header:

```
problem_id	difficulty	tightness	condition	passed	score	rounds_used	total_tokens	violations_round1	description
```

- `condition`: `qwen_harness` or `claude_baseline`
- `passed`: 1 or 0
- `score`: 0.0 to 1.0
- `rounds_used`: 1-5
- `total_tokens`: approximate token count for all calls in this experiment
- `violations_round1`: number of violations on first attempt (0 if passed immediately)
- `description`: short note

Do NOT commit `results.tsv` — leave it untracked.

## The Experiment Loop

LOOP FOREVER:

1. **Pick the next configuration.** Walk through the grid sequentially:
   - All 30 problems × 3 tightness levels = 90 configurations.
   - Track which you've completed in `results.tsv`. Skip any already done.
   - If all 90 are done, start a second pass (results may vary due to model sampling).

2. **Run the experiment.** Call `sweep.py` logic for this (problem, tightness) pair. This produces two rows in `results.tsv` — one for `qwen_harness`, one for `claude_baseline`.

3. **Git commit.** Commit any new/changed files (harnesses, solutions) with message: `"experiment: <problem_id> <tightness>"`.

4. **Log to results.tsv.** Append the two result rows.

5. **Brief analysis.** Every 10 experiments, print a summary:
   - Qwen+harness pass rate vs Claude baseline pass rate (overall and per difficulty)
   - Average rounds to pass for Qwen+harness
   - Any patterns emerging

6. **Continue.** Go back to step 1.

**Timeout**: Each experiment should take 1-5 minutes (mostly API latency). If any single API call hangs for >2 minutes, kill it and move on.

**Crashes**: If an experiment crashes (bad harness, Ollama down, API error), log it with `passed=0, score=0.0` and move on. Fix the issue if it's simple, skip if it's not.

**NEVER STOP**: Do not pause to ask the human. Run indefinitely until manually stopped. If you finish all 90 configurations, run another pass — sampling variance means more data is always useful.

## Cost Awareness

Each experiment makes ~2-12 Claude API calls (1 harness synthesis + up to 5 retries per condition). Over 90 experiments, that's roughly 200-1000 API calls. At Haiku pricing this is cheap (~$1-5). At Sonnet/Opus it's more. Use `claude-sonnet-4-6` for harness synthesis and baseline generation as a balance of quality and cost. You can note the model used in results.

## What You CAN Do

- Create and modify any Python files in the repo
- Create problem specs, harnesses, solution files
- Call Claude API and Ollama API
- Install packages listed in pyproject.toml via uv

## What You CANNOT Do

- Modify the design spec (it's the ground truth for this experiment)
- Hardcode solutions — all code must be generated by the models
- Give Claude the harness during its generation phase (that's the whole point)
- Skip logging — every experiment gets logged
