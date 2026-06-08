# Priroda

Priroda is a step-through debugger for Rust programs running under Miri.

Current focus:

- simple CLI prototype
- single-threaded stepping with Miri's interpreter
- source-location output after stepping
- source-location breakpoint prototype

## Setup

From `miri/`, install the pinned toolchain and the local `cargo-miri`
command:

```sh
./miri toolchain
./miri install
```

Then build the Miri sysroot and export it for Priroda:

```sh
cargo +miri miri setup
export MIRI_SYSROOT="$(cargo +miri miri setup --print-sysroot)"
```

## Run

Priroda currently reads `MIRI_SYSROOT` directly. After setup, run Priroda
from `miri/priroda/`:

```sh
cargo run -- ../tests/pass/empty_main.rs
```

## Test

Priroda's CLI tests also need `MIRI_SYSROOT`. Run them from `miri/priroda/`:

```sh
cargo test
```

If the CLI tests fail due to mismatched output, you can update the expected output files by running the tests with the `--bless` flag:

```sh
cargo test -- --bless
```

or 

```sh
RUSTC_BLESS=1 cargo test
```

## Commands

| Command | Description |
|---|---|
| Enter, `si`, `stepi` | Execute one Miri interpreter step. |
| `c`, `continue` | Continue until the program finishes or reaches a breakpoint. |
| `b <path>:<line>`, `break <path>:<line>` | Add a source-location breakpoint. |
| `q`, `quit` | Exit Priroda. |

EOF also exits Priroda cleanly.

Example:

```text
(priroda) break tests/pass/empty_main.rs:3
(priroda) continue
```
