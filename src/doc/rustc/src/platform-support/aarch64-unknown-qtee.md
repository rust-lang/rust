# aarch64-unknown-qtee

**Tier: 3**

Target support for the Qualcomm Trusted Execution Environment [QTEE](https://docs.qualcomm.com/doc/80-88500-4/topic/77_TrustZone_and_secure_application.html), which runs
Trusted Applications (TAs) in ARM's TrustZone.

## Target maintainers

- [@atmartorana](mailto:amartora@qti.qualcomm.com)
- [Gaurav Kashyap](mailto:gaurkash@qti.qualcomm.com)
- [Neal Haas](mailto:nhaas@qti.qualcomm.com)
- As a fallback rustzone@qti.qualcomm.com

## Requirements

This target is cross-compiled, and there is no host-tools support.

QTEE support for the standard library is a work in progress: `alloc` will be fully supported and `std` will have support that excludes
things like network I/O, spawning processes / threads (as QTEE is single threaded).
Filesystem access is limited to QTEE's Secure File System (SFS) rather than a general-purpose fileystem.
File descriptors are supported for the purpose of IPC. I/O is limited to standard output and standard error logging.

QTEE uses the ELF format. Each TA is linked as a position-dependent executable and must export `TA_APP_NAME` symbol which
identifies it to the QTEE loader.

## Setup

A copy of the QTEE SDK and built artifacts will be needed for runtime support.

## Building the target

Rust does not yet ship pre-compiled artifacts for this target. To compile you will either need to
build Rust with the target enabled (see example below), or use `build-std` to compile `std`
from source yourself.

To build a Rust toolchain, create a `bootstrap.toml` with following contents as an example:

  ```toml
  [build]
  target = ["x86_64-unknown-linux-gnu", "aarch64-unknown-qtee"]

  [target.aarch64-unknown-qtee]
  profiler = false
  sanitizers = false
  cc = "/path/to/llvm/bin/clang"
  cxx = "/path/to/llvm/bin/clang++"
  linker = "/path/to/llvm/bin/clang"
  ar = "/path/to/llvm/bin/llvm-ar"
  ranlib = "/path/to/llvm/bin/llvm-ranlib"

The upstream LLVM build packaged with Rust is sufficient to compile for the target.

## Building Rust programs

Compiling binaries for this target requires some additional linker inputs supplied from the QTEE SDK,
primarily:

- `libcmnlib.so`
- `applib.lib`
- `common_applib.o`

The directory containing these objects can optionally be set when building the Rust toolchain
for the target itself. For example setting the directory containing these objects from the prebuilts
directory of the QTEE SDK, via the `qtee-dist-path` target option in `bootstrap.toml`:

```toml
[target.aarch64-unknown-qtee]
qtee-dist-path = "/path/to/qteesdk/prebuilts/x86-64/"

Building the `aarch64-unknown-qtee` compiler and standard library does not require these objects
at all - they are only consumed when self-contained linking copies them into the target's lib
directory for an actual TA binary. If `qtee-dist-path` is unset this copy step is skipped
and the toolchain will still build. However, compiling a TA will then fail at the linker stage.

To build executables and on-target tests:

```shell
$ rustc --target aarch64-unknown-qtee your-code.rs
```

## Testing

Currently there is no support to run the rustc test suite for this target. This
is majorly due to the fact we enforce symbols like `TA_APP_NAME` to be exported
that just cannot happen when we link test files.
