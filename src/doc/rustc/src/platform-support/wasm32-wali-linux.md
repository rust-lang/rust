# `wasm32-wali-linux-*`

**Tier: 3**

WebAssembly targets that use the [WebAssembly Linux Interface (WALI)](https://github.com/arjunr2/WALI) with 32-bit memory. The latest status of the WALI specification and support are documented within the repo.

WALI offers seamless targetability of traditional Linux applications to Wasm by exposing Linux syscalls strategically into the sandbox. Numerous applications and build system work unmodified over WALI, including complex low-level system libraries -- a list of applications are included in the research paper linked in the main repo.

From the wider Wasm ecosystem perspective, implementing WALI within engines allows layering of high-level security policies (e.g. WASI) above it, arming the latter's implementations with sandboxing and portability.

## Target maintainers

[@arjunr2](https://github.com/arjunr2)

## Requirements

### Compilation
This target is cross-compiled and requires an installation of the [WALI sysroot](https://github.com/Wasm-Thin-Kernel-Interfaces/WALI.git). This produces standard `wasm32` binaries with the WALI interface methods as module imports that need to be implemented by a supported engine (see the  "Execution" section below).

`wali` targets *minimally require* the following LLVM feature flags:

* [Bulk memory] - `+bulk-memory`
* Mutable imported globals - `+mutable-globals`
* [Sign-extending operations] - `+sign-ext`
* [Threading/Atomics] - `+atomics`

[Bulk memory]: https://github.com/WebAssembly/spec/blob/main/proposals/bulk-memory-operations/Overview.md
[Sign-extending operations]: https://github.com/WebAssembly/spec/blob/main/proposals/sign-extension-ops/Overview.md
[Threading/Atomics]: https://github.com/WebAssembly/threads/blob/main/proposals/threads/Overview.md

> **Note**: Users can expect that new enabled-by-default Wasm features for LLVM are transitively incorporatable into this target -- see [wasm32-unknown-unknown](wasm32-unknown-unknown.md) for detailed information on WebAssembly features.


> **Note**: The WALI ABI is *not identical* to the `wasm32-wasip2` or `wasm32-unknown-unknown` ABI. The primary difference is 64-bit `long` types as opposed to 32-bit for wasm32. This is required to maximize portability with minimum source code changes for currently supported 64-bit host platforms. These ABIs may converge in the future as the spec evolves.

### Execution
Running generated WALI binaries also requires a supported compliant engine implementation -- a working implementation in the [WebAssembly Micro-Runtime (WAMR)](https://github.com/arjunr2/WALI) is included in the repo.

> **Note**: WALI is still somewhat experimental and bugs may exist in the Rust support, WALI toolchain, or the LLVM compiler. The former can be filed in Rust repos while the latter two in the WALI repo.

## Building the target

You can build Rust with support for the target by adding it to the `target`
list in `bootstrap.toml`, and pointing to the toolchain artifacts from the previous section ("Requirements->Compilation"). A sample `bootstrap.toml` for the `musl` environment will look like this, where `<WALI-root>` is the absolute path to the root directory of the [WALI repo](https://github.com/arjunr2/WALI):

```toml
[build]
target = ["wasm32-wali-linux-musl"]

[target.wasm32-wali-linux-musl]
musl-root = "<WALI>/build/sysroot"
cc = "<WALI>/build/llvm/bin/clang"
cxx = "<WALI>/build/llvm/bin/clang++"
ar = "<WALI>/build/llvm/bin/llvm-ar"
ranlib = "<WALI>/build/llvm/bin/llvm-ranlib"
llvm-libunwind = "system"
crt-static = true
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar (with the appropriate sysroot links).

Note that the following `cfg` directives are set for `wasm32-wali-linux-*`:

* `cfg(target_arch = "wasm32")`
* `cfg(target_family = {"wasm", "unix"})`
* `cfg(target_r = "wasm")`
* `cfg(target_os = "linux")`
* `cfg(target_env = *)`

### Restrictions

Hardware or platform-specific support, besides `syscall` is mostly unsupported in WALI for ISA portability (these tend to be uncommon).

## Testing

Currently testing is not supported for `wali` targets and the Rust project doesn't run any tests for this target.

However, standard ISA-agnostic tests for Linux should be thereotically reusable for WALI targets and minor changes. Testing integration will be continually incorporated as support evolves.


## Cross-compilation toolchains and C code

Most fully featured C code is compilable with the WALI toolchain -- examples can be seen in the repo.
