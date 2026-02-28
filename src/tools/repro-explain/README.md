# repro-explain

`repro-explain` is an in-tree tool for investigating Rust build reproducibility diffs.  
It compares two build runs, classifies differing artifacts, collects evidence, runs confirmation experiments, and generates reports.

## What It Does

- Runs a build twice and analyzes diffs automatically (`run-twice`)
- Compares and analyzes two existing runs (`diff` + `explain`)
- Produces semantic diffs for changed artifacts
- Classifies likely root-cause categories
- Estimates the `first divergent stage` (stage replay)
- Generates HTML and JSON reports

Main diff classes:

- `path-leak`
- `timestamp`
- `env-leak`
- `unstable-order`
- `build-script`
- `proc-macro`
- `metadata-stage`
- `codegen-stage`
- `link-stage`
- `schedule-sensitive-parallelism`
- `unknown`

## Build and Run

Run from the workspace root (`rust/`):

```bash
cargo run -p repro-explain -- --help
```

Inside the Rust repository, you can also run it through `x`:

```bash
./x run src/tools/repro-explain -- --help
```

## Quick Start

Analyze a Cargo workspace by running the build twice:

```bash
cargo run -p repro-explain -- run-twice -- cargo build --release
```

Analyze an `x.py` build by running it twice:

```bash
cargo run -p repro-explain -- run-twice -- ./x build --stage 0 compiler/rustc
```

Analyze existing runs:

```bash
cargo run -p repro-explain -- capture --run-id A -- cargo build --release
cargo run -p repro-explain -- capture --run-id B -- cargo build --release
cargo run -p repro-explain -- diff --left .repro/runs/A --right .repro/runs/B
cargo run -p repro-explain -- explain --analysis .repro/analysis/A__B
cargo run -p repro-explain -- report --analysis .repro/analysis/A__B
```

## Subcommands

- `run-twice -- <command...>`
- `capture --run-id <id> -- <command...>`
- `diff --left <run_dir> --right <run_dir>`
- `explain --analysis <dir> [--artifact <glob>] [--confirm <level>]`
- `report --analysis <dir>`

## `--confirm` Behavior

Current implementation behavior:

- `none`
  - No extra replay
- `cheap`
  - For `timestamp` suspects: replay twice with `SOURCE_DATE_EPOCH=1700000000`
  - For `unstable-order` / `schedule-sensitive-parallelism` suspects: replay with `--jobs 1`
- `standard`
  - Includes `cheap`, plus stage replay
  - Compares `metadata -> mir -> llvm-ir -> obj` to find the first divergent stage
- `full`
  - Includes `standard`, plus build script replay
  - Runs same-source-dir replay for path-leak confirmation only when `--same-source-replay` is set

## Output Directory

Default is `.repro/` (configurable via `--work-dir`).

### Run Data

`.repro/runs/A` (same layout for `B`):

- `meta.json`
- `command.json`
- `cargo-metadata.json` (when using Cargo)
- `cargo-messages.jsonl`
- `compiler-artifacts.json`
- `build-script-executed.json`
- `build-script-stdout.json`
- `invocations-rustc.jsonl`
- `invocations-rustdoc.jsonl`
- `artifacts.json`
- `hashes.json`
- `out-dir-manifest.json`
- `stdout.log`, `stderr.log`
- `timings/` (copied from `target/cargo-timings` when present)
- `target/`

### Analysis Data

`.repro/analysis/A__B`:

- `diff-manifest.json`
- `provenance.json`
- `findings.json`
- `report.html`
- `artifacts/<artifact-id>/...`
  - `semantic-diff.json`
  - `semantic-diff.txt`
  - `rule-hits.json`
  - `evidence.json`
  - `finding.json`
  - `stage-localization.json` (when generated)

## Reading the Results

For each artifact, `findings.json` and the HTML report provide:

- `status`: `confirmed` / `strong-suspect` / `weak-suspect`
- `class`: diff class
- `first_divergent_stage`: earliest observed divergence stage
- `primary_locus`: candidate file:line
- `evidence`: semantic diff / replay / rule-hit evidence
- `fix_hint`: suggested fix direction

## Environment Variable Capture

By default, environment capture uses an allowlist.  
Use `--capture-all-env` only when full environment capture is needed.

## Notes and Limitations

- `diffoscope` is optional; fallback backends are used if it is unavailable.
- Partial capture is preserved even when the build fails.
- For non-Cargo commands (for example `x.py`), collection is primarily wrapper-based.
- Replay can be expensive, especially with `full` + `--same-source-replay`.

## For Developers

Run tests:

```bash
cargo test -p repro-explain
```
