# Priroda

Priroda is a step-through debugger for Rust programs running under Miri.

Current focus:

- simple CLI prototype
- single-threaded stepping with Miri's interpreter
- source-location output after stepping
- source-location breakpoint prototype
- source-local listing prototype
- runtime local state and value rendering
- range-limited byte output for indirect locals

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
| `s`, `step` | Step until the displayed source location changes. |
| `c`, `continue` | Continue until the program finishes or reaches a breakpoint. |
| `b <path>:<line>`, `break <path>:<line>` | Add a source-location breakpoint. |
| `l`, `locals` | List source-level locals in the current frame by name. |
| `p <local>`, `print <local>` | Print one MIR local by numeric id. |
| `q`, `quit` | Exit Priroda. |

## Value Output

Immediate values use Miri's `Immediate` display representation. Indirect
locals are rendered as the bytes belonging to the current value range, not as
the entire backing allocation:

```text
[01 02 03]
[?? ?? ??]
```

`??` means the byte is uninitialized. A value whose runtime size cannot be
determined is reported as `<unsupported-unsized>`.

Pointer/provenance spans are planned as part of the raw byte output, using a
compact dump-like marker such as:

```text
[<ptr alloc5+0> 2a 00 00 00]
```

Automatic pointer following is future work and should be explicit, not part of
ordinary value printing. Typed field rendering and dereference/projection-aware
printing are also future work.

EOF also exits Priroda cleanly.

Example:

```text
(priroda) break tests/pass/empty_main.rs:3
(priroda) continue
```
