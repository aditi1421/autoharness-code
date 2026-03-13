# AutoHarness Stress Test: Design Spec

## Research Question

Does a smaller model (Qwen-2.5-Coder 1.5B) + auto-synthesized constraint harness beat a larger model (Claude) without a harness, in numerical code generation?

Based on: [AutoHarness (Lou et al., arXiv 2603.03329)](https://arxiv.org/abs/2603.03329) — which showed Gemini Flash + self-generated harness beats Gemini Pro in TextArena games.

## Core Comparison

| Condition | Model | Harness | Retry with feedback |
|-----------|-------|---------|-------------------|
| Baseline | Claude (API) | None | Yes, but no constraint feedback |
| Treatment | Qwen-2.5-Coder 1.5B (Ollama) | Claude-synthesized | Yes, harness violation feedback |

Both conditions get the same retry budget (5 rounds). The difference: the small model gets structured constraint feedback from the harness on each failure. Claude gets only "try again."

## Domain: Numerical/Mathematical Code Generation

### Problem Benchmark

~30 problems across 3 difficulty levels (10 each):

**Easy:** dot product, matrix transpose, polynomial evaluation, GCD, Euclidean distance, vector normalization, running mean, power function, modular exponentiation, linear interpolation

**Medium:** matrix inverse, FFT, numerical integration (Simpson's rule), solving linear systems (Gaussian elimination), matrix determinant, Cholesky decomposition, Newton's method root finding, cubic spline interpolation, convolution, LU decomposition

**Hard:** SVD, eigenvalue decomposition, ODE solver (RK4), sparse matrix multiplication, conjugate gradient solver, QR decomposition, Lanczos algorithm, numerical Laplace transform, multivariate Newton's method, pseudoinverse

Each problem is a JSON spec:
```json
{
  "id": "matrix_inverse",
  "difficulty": "medium",
  "description": "Implement a function that computes the inverse of an NxN matrix",
  "function_signature": "def matrix_inverse(m: list[list[float]]) -> list[list[float]]",
  "constraints": [
    "must handle singular matrices by raising ValueError",
    "numerical precision within 1e-6"
  ],
  "reference_solution": "..."
}
```

## Harness Synthesis

Claude reads a problem spec and generates a constraint harness — Python code that validates the solution.

### Harness components:
1. **Output validation** — type checking, shape checking, range bounds
2. **Property tests** — e.g., `A @ inverse(A) ~ identity`, commutativity, symmetry
3. **Edge cases** — singular matrices, empty inputs, extreme values, zero vectors
4. **Reference comparison** — compares against NumPy/SciPy reference implementation

### Harness tightness levels (swept as experiment variable):
- **Minimal** — only type/shape validation
- **Standard** — type/shape + property tests + basic edge cases
- **Strict** — all of the above + exhaustive random inputs + reference comparison

### Harness interface:
```python
def harness(fn) -> tuple[bool, float, list[str]]:
    """Returns (passed, score 0-1, list of violation descriptions)"""
```

## Experiment Pipeline

For each configuration `(problem, harness_tightness)`:

### Small model + harness path:
1. Claude generates harness from problem spec (at specified tightness)
2. Qwen 1.5B generates implementation
3. Harness runs -> violations + score
4. If violations: feed violation list back to Qwen, it retries
5. Repeat up to 5 rounds or until pass
6. Log results

### Claude baseline path:
1. Claude generates implementation (no harness during generation)
2. Evaluate using the **same harness** as the small model path (shared evaluation)
3. If fail: Claude retries with generic "tests failed" feedback (no constraint details)
4. Repeat up to 5 rounds or until pass
5. Log results

### Important: shared evaluation
Both conditions are scored by the same harness after generation. The difference is only in what feedback the model receives during retries — Qwen gets violation details, Claude gets "tests failed."

## Autoresearch Integration

### Orchestration
Single `program.md` drives an autonomous overnight loop, identical to existing autoresearch pattern:
- Agent loops through configurations sequentially
- Each configuration is one "experiment"
- Results logged to `results.tsv`
- Git branch per run (`autoharness/<date>`)
- Agent runs indefinitely until manually stopped

### Sweep variables:
- Problem difficulty: easy / medium / hard
- Harness tightness: minimal / standard / strict
- Total configurations per sweep: 30 problems x 3 tightness levels = 90 experiments
- Estimated time: ~2-5 min per experiment (API calls + Ollama inference) = ~3-7 hours

## Project Structure

```
autoharness-code/
  program.md          — agent instructions (the autonomous loop)
  problems/           — JSON problem specs (30 files)
  harnesses/          — generated constraint harnesses (cached)
  solutions/          — generated code from both models
  evaluate.py         — runs harness against solution, scores it
  generate.py         — calls Claude API + Ollama API
  sweep.py            — orchestrates the full experiment
  results.tsv         — experiment log
  analysis.ipynb      — plots and comparison
  pyproject.toml      — dependencies
  README.md           — project overview
```

## Metrics

| Metric | Description |
|--------|-------------|
| Pass rate | % of problems solved, per condition per difficulty |
| Rounds to pass | How many retries needed (small model + harness) |
| Token cost | Total tokens used per problem, per condition |
| Violations per round | How many constraint violations caught per retry |
| Score curve | Score improvement across retry rounds |

## Key Dependencies

- `anthropic` — Claude API for harness synthesis + baseline code gen
- `ollama` — local Qwen 2.5 Coder 1.5B serving (set `num_ctx=8192` to handle retry context)
- `numpy`, `scipy` — reference solutions for validation
- `uv` — package management (consistent with autoresearch)

## Notes for Future Extensions

- **Condition C (Claude + harness):** Adding Claude with the same harness feedback would isolate whether gains come from the harness itself vs model capability. Not needed to answer the core question, but valuable for a follow-up.
- **Adversarial loop (Phase 2):** A separate adversary agent that reads code and crafts targeted breaking inputs, tightening the harness over rounds.

## Success Criteria

The experiment succeeds (confirms AutoHarness thesis) if:
- Qwen 1.5B + strict harness pass rate > Claude alone pass rate on at least one difficulty level
- OR: Qwen 1.5B + harness achieves comparable pass rate at significantly lower token cost
- Stricter harnesses correlate with higher pass rates for the small model
