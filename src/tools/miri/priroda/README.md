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

## DAP Prototype

Priroda speaks a bounded Debug Adapter Protocol prototype over stdio with
`--dap`, or over TCP with `--port N`. It currently supports the startup
handshake, stops at the first
user-relevant source location after `configurationDone`, reports one current
stack frame, exposes one flat Locals scope, and maps `list_locals()` into DAP
variables with no child expansion.

The `next` and `stepIn` requests are wired to Priroda's existing source-line
step so VS Code can drive one visible step. They are not true DAP step-over or
step-in semantics yet.

### VS Code

VS Code can start Priroda as a TCP DAP server and then attach to that server
when you run the debugger configuration. The launch configuration does not spawn
Priroda directly; it starts a background task and then connects through
`debugServer`.

This requires a VS Code debug extension that contributes the `priroda` debugger
type. The `debugServer` setting only tells VS Code to connect to an
already-running adapter; it does not register a new debugger type. On a clean VS
Code install, copying these JSON files is not enough for the launch
configuration to be accepted.

After that debugger type is registered, copy the example files into the
workspace you want to debug:

```sh
mkdir -p /path/to/project/.vscode
cp vscode_launch.json /path/to/project/.vscode/launch.json
cp vscode_tasks.json /path/to/project/.vscode/tasks.json
```

Before running the debugger configuration, make sure:

- Priroda has been built, so the binary path in `command` exists.
- `MIRI_SYSROOT` points at a Miri sysroot, for example from
  `cargo +miri miri setup --print-sysroot`.
- If running the `priroda` binary directly, `LD_LIBRARY_PATH` may need to point
  at the pinned `miri` toolchain's `lib` directory.
- The Rust file path at the end of `args` is the file you want Priroda to run.
- Port `4711` is free, or both `--port` and `debugServer` use the same different
  port.
- VS Code has a debugger contribution installed that accepts
  `type: "priroda"` debug configurations.

Then edit `.vscode/tasks.json` for your local paths. Set `command` to the
Priroda binary you want VS Code to run:

```json
"command": "${workspaceFolder}/target/debug/priroda"
```

If the binary cannot find rustc libraries, add an `env` block under
`options`:

```json
"options": {
    "cwd": "${workspaceFolder}",
    "env": {
        "LD_LIBRARY_PATH": "/path/to/miri-toolchain/lib",
        "MIRI_SYSROOT": "/path/to/miri-sysroot"
    }
}
```

Also edit the final argument in `args` to point at the Rust file you want
Priroda to run. This task argument, not `launch.json`, selects the interpreted
program:

```json
"${workspaceFolder}/src/main.rs"
```

The task runs Priroda like this:

```sh
cargo run -- --dap --port 4711 /path/to/project/src/main.rs
```

Once Priroda prints `priroda dap listening on 127.0.0.1:4711`, VS Code treats
the background task as ready and connects with:

```json
{
    "type": "priroda",
    "request": "launch",
    "preLaunchTask": "Priroda: Start DAP Server",
    "debugServer": 4711
}
```

Priroda accepts one TCP connection and waits for VS Code before running the DAP
handshake.

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
| `f <alloc> <offset>`, `follow <alloc> <offset>` | Render allocation bytes from an offset, including the full allocation size. |
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
