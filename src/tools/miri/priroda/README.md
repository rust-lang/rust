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

DAP supports `stepIn`, `next`, and `stepOut`. `stepIn` stops at the next
displayed source location and can enter calls. `next` steps over calls by
tracking the starting stack depth, and `stepOut` runs until execution reaches a
shallower stack frame. This is still single-threaded and source-position based,
not the full future thread/frame model.

### VS Code

VS Code can start Priroda as a TCP DAP server and then attach to that server
when you run the debugger configuration. The launch configuration does not spawn
Priroda directly; it starts a background task and then connects through
`debugServer`.

`debugServer` tells VS Code to connect to an already-running adapter, but VS Code
still requires the launch configuration's `type` to be one it knows. Priroda has
no installed extension, so the configuration uses VS Code's built-in `node`
debug type as the registered editor-side type. `debugServer` redirects the DAP
transport to Priroda before the Node adapter is spawned, so no custom Priroda
extension is needed. The configuration uses `request: "attach"`, which makes VS
Code send the DAP `attach` request; Priroda accepts `attach` as the same startup
transition as `launch`. A richer custom Priroda extension is deferred to future
graphical features.

The templates assume `${workspaceFolder}` is the `miri/priroda` directory that
contains them. Copy them into that directory's `.vscode/`, or edit
`--manifest-path` and the final `args` entry when using them from elsewhere:

```sh
mkdir -p /path/to/miri/priroda/.vscode
cp vscode_launch.json /path/to/miri/priroda/.vscode/launch.json
cp vscode_tasks.json /path/to/miri/priroda/.vscode/tasks.json
```

Before running the debugger configuration, make sure:

- The Rust file path at the end of `args` is the file you want Priroda to run.
- Port `4711` is free, or both `--port` and `debugServer` use the same different
  port.
- `MIRI_SYSROOT` points at a Miri sysroot, for example from
  `cargo +miri miri setup --print-sysroot`. VS Code resolves `${env:MIRI_SYSROOT}`
  from the environment it was started with, not from the task's `env`, so export
  it in your shell before launching VS Code, or replace the argument with the
  absolute sysroot path.
- The `cargo` in `command` resolves to the `miri` toolchain's cargo, so the task
  builds Priroda with `rustc_private`. That is automatic when the workspace is
  `miri/priroda`; when using the templates from another project, pass `+miri` as
  the first `cargo` argument.

The task runs Priroda through `cargo run` against the Priroda crate:

```sh
cargo run --manifest-path /path/to/miri/priroda/Cargo.toml -- \
    --dap --port 4711 --sysroot "$MIRI_SYSROOT" /path/to/project/src/main.rs
```

Edit the final argument in `args` to point at the Rust file you want Priroda to
run (the checked-in default is `../tests/pass/empty_main.rs`). This task
argument, not `launch.json`, selects the interpreted program.

Running through `cargo run` sets the dynamic library path automatically. The
`priroda` binary links rustc's shared libraries, so running it directly, or via
`cargo install`, still needs `LD_LIBRARY_PATH` to point at the pinned `miri`
toolchain's `lib` directory; a future packaging step (an rpath, or shipping
Priroda next to Miri) will remove that requirement.

Once Priroda prints `priroda dap listening on 127.0.0.1:4711`, VS Code treats
the background task as ready and connects with:

```json
{
    "type": "node",
    "request": "attach",
    "preLaunchTask": "Priroda: Start DAP Server",
    "debugServer": 4711
}
```

Priroda accepts one TCP connection and waits for VS Code before running the DAP
handshake. VS Code's built-in JavaScript debugger may also send extension
requests of its own, such as `enableNetworking` for its network preview; Priroda
skips unrecognized requests rather than failing, so those are ignored.

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
| `s`, `step` | Step to the next displayed source location, entering calls. |
| `n`, `next` | Step over the current displayed source location. |
| `out`, `stepout` | Run until execution returns to a shallower stack frame. |
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
