# `wasm32-wasip1`

**Tier: 2**

The `wasm32-wasip1` target is a WebAssembly compilation target which
assumes that the [WASIp1] (aka "WASI preview1") set of "syscalls" are available
for use in the standard library. Historically this target in the Rust compiler
was one of the first for WebAssembly where Rust and C code are explicitly
intended to interoperate as well.

There's a bit of history to the target and current development which is also
worth explaining before going much further. Historically this target was
originally called `wasm32-wasi` in both rustc and Clang. This was first added
to Rust in 2019. In the intervening years leading up to 2024 the WASI standard
continued to be developed and was eventually "rebased" on top of the [Component
Model]. This was a large change to the WASI specification and was released as
0.2.0 ("WASIp2" colloquially) in January 2024. The previous target's name in
rustc, `wasm32-wasi`, was then renamed to `wasm32-wasip1`, to avoid
confusion with this new target to be added to rustc as `wasm32-wasip2`.
Some more context can be found in these MCPs:

* [Rename wasm32-wasi target to wasm32-wasip1](https://github.com/rust-lang/compiler-team/issues/607)
* [Smooth the renaming transition of wasm32-wasi](https://github.com/rust-lang/compiler-team/issues/695)

At this point the `wasm32-wasip1` target is intended for historical
compatibility with the first version of the WASI standard. As of now (January
2024) the 0.2.0 target of WASI ("WASIp2") is relatively new. The state of
WASI will likely change in few years after which point this documentation will
probably receive another update.

[WASI Preview1]: https://github.com/WebAssembly/WASI/tree/main/legacy/preview1
[Component Model]: https://github.com/webassembly/component-model

Today the `wasm32-wasip1` target will generate core WebAssembly modules
which will import functions from the `wasi_snapshot_preview1` module for
OS-related functionality (e.g. printing).

## Target maintainers

When this target was added to the compiler platform-specific documentation here
was not maintained at that time. This means that the list below is not
exhaustive and there are more interested parties in this target. That being
said since when this document was last updated those interested in maintaining
this target are:

- Alex Crichton, https://github.com/alexcrichton

## Requirements

This target is cross-compiled. The target includes support for `std` itself,
but not all of the standard library works. For example spawning a thread will
always return an error (see the `wasm32-wasip1-threads` target for
example). Another example is that spawning a process will always return an
error. Operations such as opening a file, however, will be implemented by
calling WASI-defined APIs.

The WASI targets for Rust are explicitly intended to interoperate with other
languages compiled to WebAssembly, for example C/C++. Any ABI differences or
mismatches are considered bugs that need to be fixed.

By default the WASI targets in Rust ship in rustup with a precompiled copy of
[`wasi-libc`] meaning that a WebAssembly-targetting-Clang is not required to
use the WASI targets from Rust.  If there is no actual interoperation with C
then `rustup target add wasm32-wasip1` is all that's needed to get
started with WASI.

Note that this behavior can be controlled with `-Clinker` and
`-Clink-self-contained`, however. By specifying `clang` as a linker and
disabling the `link-self-contained` option an external version of `libc.a` can
be used instead.

[`wasi-libc`]: https://github.com/WebAssembly/wasi-libc

## Building the target

To build this target a compiled version of [`wasi-libc`] is required to be
present at build time. This can be installed through
[`wasi-sdk`](https://github.com/WebAssembly/wasi-sdk) as well. This is the
configured with:

```toml
[target.wasm32-wasip1]
wasi-root = ".../wasi-libc/sysroot"
```

Additionally users will need to enable LLD when building Rust from source as
LLVM's `wasm-ld` driver for LLD is required when linking WebAssembly code
together.

## Building Rust programs

The `wasm32-wasip1` target is shipped with rustup so users can install
the target with:

```text
rustup target add wasm32-wasip1
```

> **Note**: the `wasm32-wasip1` target is new and may only be available
> on nightly by the time you're reading this. If `wasm32-wasip1` isn't
> available on stable Rust then `wasm32-wasi` should be available instead.

Rust programs can be built for that target:

```text
rustc --target wasm32-wasip1 your-code.rs
```

## Cross-compilation

This target can be cross-compiled from any hosts.

## Testing

Currently the WASI target is not tested in rust-lang/rust CI. This means that
tests in the repository are not guaranteed to pass. This is theoretically
possibly by installing a standalone WebAssembly runtime and using it as a
"runner" for all tests, but there are various failures that will need to be
waded through to adjust tests to work on the WASI target.
