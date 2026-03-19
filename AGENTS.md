# AGENTS.md

Small repos forget quickly; this file is the memory aid.

## Project Snapshot
- Repo workflow: this is an `openai/parameter-golf` fork setup, with local work typically pushed to the fork and upstream left alone unless explicitly requested.
- Local environment: Windows x64, Intel CPU, no CUDA GPU, no Apple Silicon.
- Current local smoke path: `train_gpt_cpu_smoke.py`.
- The smoke script is for local verification and observability only, not real challenge training or a final submission path.

## Git / Branch Rules
- Never work directly on `main` unless explicitly told otherwise.
- Prefer small feature or experiment branches.
- Make one measurable change per step.
- Show a diff summary before commit.
- Small verified changes may be committed and pushed by default.
- Stop before push only if there is a significant error, failed local verification, ambiguous or risky scope, or something clearly off-track.
- Keep checkpoint commits small and use clear, literal commit messages.
- If a push turns out wrong, prefer a clean follow-up fix or revert rather than panic edits.

## Change Strategy
- Prefer observability and understanding improvements before adding complexity.
- Keep edits minimal, local, and reversible.
- Do not bundle multiple improvements into one step.
- Verify locally after changes when possible.
- Report the exact commands run and the visible output changes.

## Context Preservation
- End each finished step with a 3-line checkpoint summary:
  1. what changed
  2. why
  3. next safest step
- After context loss, first read `AGENTS.md`, `git status --short --branch`, `git log --oneline -n 10`, and the relevant script before editing.
- Prefer durable repo state over long chat memory.

## Repo-Specific Notes
- Do not treat `train_gpt_cpu_smoke.py` as a final submission path.
- Keep CPU smoke work small and isolated; the main reference scripts remain `train_gpt.py` and `train_gpt_mlx.py`.
- Do not commit `logs/`, `data/datasets/`, `data/tokenizers/`, checkpoints, or `.venv/`.
- Preserve the existing remotes and branch structure unless explicitly told otherwise.

## Useful Commands
Typical local loop:

```powershell
.\.venv\Scripts\python.exe .\train_gpt_cpu_smoke.py
git status --short --branch
git log --oneline -n 10
git push
```

## Boundaries
- Do not rewrite this file gratuitously.
- Update it only when the workflow meaningfully changes.
