# REMOTE_RUN_CHECKLIST.md

## Purpose
- First remote CUDA baseline run checklist for Parameter Golf.
- Written for this repo and this fork-based workflow, not generic cloud-GPU advice.

## Preconditions Before Renting GPU Time
- Branch is pushed to your fork and the local repo is clean.
- Record the exact commit hash you intend to run.
- Expected baseline data path: `./data/datasets/fineweb10B_sp1024/`
- Expected baseline tokenizer path: `./data/tokenizers/fineweb_1024_bpe.model`
- Expected baseline vocab: `VOCAB_SIZE=1024`
- Confirm OpenAI compute grant / Runpod billing is actually available before launching a pod.
- Confirm your Runpod SSH key is configured so you can get into the box immediately.

## Remote Baseline Steps
### Runpod UI
- Create or open the Runpod account.
- Start with a 1xH100 pod, not 8xH100.
- Use the official Parameter Golf template from the README.
- Enable SSH terminal access.
- Wait for the pod to become reachable, then SSH into `/workspace/`.

### After SSH Login
- Clone your fork so the exact pushed branch/commit is available remotely.
- Check out the branch or commit you recorded before renting the GPU.
- Use the preinstalled environment in the template image.
- Download the cached FineWeb SP1024 dataset.
- Run the baseline command exactly:

```bash
cd /workspace
git clone https://github.com/nomin132/parameter-golf.git
cd parameter-golf
git checkout <branch-or-commit-to-run>
python3 data/cached_challenge_fineweb.py --variant sp1024
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Success Signals
- Startup prints a log path like `logs/<RUN_ID>.txt`.
- Setup logs include `val_bpb:enabled tokenizer_kind=sentencepiece`.
- Training logs include `step:... train_loss:...`.
- Validation logs include `step:... val_loss:... val_bpb:...` at the end, and optionally during the run if `VAL_LOSS_EVERY` is set.
- Compressed model size appears in `Serialized model int8+zlib: ... bytes` and `Total submission size int8+zlib: ...`.
- Final success lines are:
  - `final_int8_zlib_roundtrip val_loss:... val_bpb:... eval_time:...`
  - `final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...`
- README baseline expectation: final `val_bpb` around `~1.2` and compressed model size under `16MB`.
- Keep after the run:
  - `logs/<RUN_ID>.txt`
  - `final_model.int8.ptz`
  - exact commit hash
  - exact launch command and env vars
  - `final_model.pt` only if you want an uncompressed debugging artifact

## Abort / Bad Signals
- `CUDA is required` or any immediate NCCL / CUDA init failure.
- Dataset or tokenizer path errors, or `VOCAB_SIZE` mismatch.
- No `train_loss` step logs after setup completes.
- Run exits before the final `final_int8_zlib_roundtrip` lines.
- Compressed artifact is obviously over budget or the run is clearly not following the baseline path you intended.

## First Post-Baseline Actions
- Record immediately:
  - commit hash
  - GPU SKU
  - exact command
  - final `val_bpb`
  - compressed model size from `Serialized model int8+zlib`
  - path to `logs/<RUN_ID>.txt`
- Copy the log and `final_model.int8.ptz` off the remote box before shutting it down.
- Safest next iteration: keep the same data/tokenizer path and make exactly one measurable change on a branch before spending more GPU time.

## Tiny Command Block
```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
