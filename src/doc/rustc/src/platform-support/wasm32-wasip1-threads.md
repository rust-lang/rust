# `wasm32-wasip1-threads`

**Tier: 2**

The `wasm32-wasip1-threads` target is a new and still (as of July 2023) an
experimental target. This target is an extension to `wasm32-wasip1` target,
originally known as `wasm32-wasi`. It extends the original target with a
standardized set of syscalls that are intended to empower WebAssembly binaries
with native multi threading capabilities.

> **Note**: Prior to March 2024 this target was known as
> `wasm32-wasi-preview1-threads`, and even longer before that it was known as
> `wasm32-wasi-threads`.

[wasi-threads]: https://github.com/WebAssembly/wasi-threads
[threads]: https://github.com/WebAssembly/threads


## Target maintainers

[@g0djan](https://github.com/g0djan)
[@alexcrichton](https://github.com/alexcrichton)
[@abrown](https://github.com/abrown)
[@loganek](https://github.com/loganek)

## Requirements

This target is cross-compiled. The target supports `std` fully.

The Rust target definition here is interesting in a few ways. We want to
serve two use cases here with this target:
* First, we want Rust usage of the target to be as hassle-free as possible,
  ideally avoiding the need to configure and install a local wasm32-wasip1-threads
  toolchain.
* Second, one of the primary use cases of LLVM's new wasm backend and the
  wasm support in LLD is that any compiled language can interoperate with
  any other. The `wasm32-wasip1-threads` target is the first with a viable C
  standard library and sysroot common definition, so we want Rust and C/C++
  code to interoperate when compiled to `wasm32-unknown-unknown`.


You'll note, however, that the two goals above are somewhat at odds with one
another. To attempt to solve both use cases in one go we define a target
that (ab)uses the `crt-static` target feature to indicate which one you're
in.
### No interop with C required
By default the `crt-static` target feature is enabled, and when enabled
this means that the bundled version of `libc.a` found in `liblibc.rlib`
is used. This isn't intended really for interoperation with a C because it
may be the case that Rust's bundled C library is incompatible with a
foreign-compiled C library. In this use case, though, we use `rust-lld` and
some copied crt startup object files to ensure that you can download the
wasi target for Rust and you're off to the races, no further configuration
necessary.
All in all, by default, no external dependencies are required. You can
compile `wasm32-wasip1-threads` binaries straight out of the box. You can't, however,
reliably interoperate with C code in this mode (yet).
### Interop with C required
For the second goal we repurpose the `target-feature` flag, meaning that
you'll need to do a few things to have C/Rust code interoperate.
1. All Rust code needs to be compiled with `-C target-feature=-crt-static`,
   indicating that the bundled C standard library in the Rust sysroot will
   not be used.
2. If you're using rustc to build a linked artifact then you'll need to
   specify `-C linker` to a `clang` binary that supports
   `wasm32-wasip1-threads` and is configured with the `wasm32-wasip1-threads` sysroot. This
   will cause Rust code to be linked against the libc.a that the specified
   `clang` provides.
3. If you're building a staticlib and integrating Rust code elsewhere, then
   compiling with `-C target-feature=-crt-static` is all you need to do.

All in all, by default, no external dependencies are required. You can
compile `wasm32-wasip1-threads` binaries straight out of the box. You can't, however,
reliably interoperate with C code in this mode (yet).


Also note that at this time the `wasm32-wasip1-threads` target assumes the
presence of other merged wasm proposals such as (with their LLVM feature flags):

* [Bulk memory] - `+bulk-memory`
* Mutable imported globals - `+mutable-globals`
* Atomics - `+atomics`

[Bulk memory]: https://github.com/WebAssembly/spec/blob/main/proposals/bulk-memory-operations/Overview.md

LLVM 16 is required for this target. The reason is related to linker flags: prior to LLVM 16, --import-memory and --export-memory were not allowed together. The reason both are needed is an artifact of how WASI currently does things; see https://github.com/WebAssembly/WASI/issues/502 for more details.

The target intends to match the corresponding Clang target for its `"C"` ABI.

> **Note**: due to the relatively early-days nature of this target when working
> with this target you may encounter LLVM bugs. If an assertion hit or a bug is
> found it's recommended to open an issue either with rust-lang/rust or ideally
> with LLVM itself.

## Platform requirements

The runtime should support the same set of APIs as any other supported wasi target for interacting with the host environment through the WASI standard. The runtime also should have implementation of [wasi-threads proposal](https://github.com/WebAssembly/wasi-threads).

This target is not a stable target. This means that there are a few engines
which implement the `wasi-threads` feature and if they do they're likely behind a
flag, for example:

* Wasmtime - `--wasi threads`
* [WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) - needs to be built with WAMR_BUILD_LIB_WASI_THREADS=1

## Building the target

Users need to install or built wasi-sdk since release 20.0
https://github.com/WebAssembly/wasi-sdk/releases/tag/wasi-sdk-20
and specify path to *wasi-root* `bootstrap.toml`

```toml
[target.wasm32-wasip1-threads]
wasi-root = ".../wasi-libc/sysroot"
```

After that users can build this by adding it to the `target` list in
`bootstrap.toml`, or with `-Zbuild-std`.

## Building Rust programs

From Rust Nightly 1.71.1 (2023-08-03) on the artifacts are shipped pre-compiled:

```text
rustup target add wasm32-wasip1-threads --toolchain nightly
```

Rust programs can be built for that target:

```text
rustc --target wasm32-wasip1-threads your-code.rs
```

## Cross-compilation

This target can be cross-compiled from any hosts.

## Testing

Currently testing is not well supported for `wasm32-wasip1-threads` and the
Rust project doesn't run any tests for this target. However the UI testsuite can be run
manually following this instructions:

0. Ensure [wamr](https://github.com/bytecodealliance/wasm-micro-runtime), [wasmtime](https://github.com/bytecodealliance/wasmtime)
or another engine that supports `wasi-threads` is installed and can be found in the `$PATH` env variable.
1. Clone master branch.
2. Apply such [a change](https://github.com/g0djan/rust/compare/godjan/wasi-threads...g0djan:rust:godjan/wasi-run-ui-tests?expand=1) with an engine from the step 1.
3. Run `./x.py test --target wasm32-wasip1-threads tests/ui` and save the list of failed tests.
4. Checkout branch with your changes.
5. Apply such [a change](https://github.com/g0djan/rust/compare/godjan/wasi-threads...g0djan:rust:godjan/wasi-run-ui-tests?expand=1) with an engine from the step 1.
6. Run `./x.py test --target wasm32-wasip1-threads tests/ui` and save the list of failed tests.
7. For both lists of failed tests run `cat list | sort > sorted_list` and compare it with `diff sorted_list1 sorted_list2`.

## Conditionally compiling code

It's recommended to conditionally compile code for this target with:

```text
#[cfg(all(target_os = "wasi", target_env = "p1", target_feature = "atomics"))]
```

Prior to Rust 1.80 the `target_env = "p1"` key was not set. Currently the
`target_feature = "atomics"` is Nightly-only. Note that the precise `#[cfg]`
necessary to detect this target may change as the target becomes more stable.

## Enabled WebAssembly features

The default set of WebAssembly features enabled for compilation includes two
more features in addition to that which
[`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md) enables:

* `bulk-memory`
* `atomics`

For more information about features see the documentation for
[`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md), but note that the
`mvp` CPU in LLVM does not support this target as it's required that
`bulk-memory`, `atomics`, and `mutable-globals` are all enabled.
