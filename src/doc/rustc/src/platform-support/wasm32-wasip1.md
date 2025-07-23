# `wasm32-wasip1`

**Tier: 2**

The `wasm32-wasip1` target is a WebAssembly compilation target which
assumes that the [WASIp1] (aka "WASI preview1") set of "syscalls" are available
for use in the standard library. This target explicitly supports interop with
non-Rust code such as C and C++.

The [WASIp1] set of syscalls is standard insofar as it was written down once by
a set of folks and has not changed since then. Additionally the [WASIp1]
syscalls have been adapted and adopted into a number of runtimes and embeddings.
It is not standard in the sense that there are no formal semantics for each
syscall and APIs are no longer receiving any maintenance (e.g. no new APIs, no
new documentation, etc). After [WASIp1] was originally developed in 2019 the
WASI standard effort has since been "rebased" on top of the [Component Model].
This was a large change to the WASI specification and was released as 0.2.0
("WASIp2" colloquially) in January 2024. Current standardization efforts are
focused on the Component Model-based definition of WASI. At this point the
`wasm32-wasip1` Rust target is intended for historical compatibility with
[WASIp1] set of syscalls.

[WASIp1]: https://github.com/WebAssembly/WASI/tree/main/legacy/preview1
[Component Model]: https://github.com/webassembly/component-model

Today the `wasm32-wasip1` target will generate core WebAssembly modules
which will import functions from the `wasi_snapshot_preview1` module for
OS-related functionality (e.g. printing).

> **Note**: Prior to March 2024 this target was known as `wasm32-wasi` with some
> historical context found in old MCPs:
>
> * [Rename wasm32-wasi target to wasm32-wasip1](https://github.com/rust-lang/compiler-team/issues/607)
> * [Smooth the renaming transition of wasm32-wasi](https://github.com/rust-lang/compiler-team/issues/695)

## Target maintainers

When this target was added to the compiler platform-specific documentation here
was not maintained at that time. This means that the list below is not
exhaustive and there are more interested parties in this target. That being
said since when this document was last updated those interested in maintaining
this target are:

[@alexcrichton](https://github.com/alexcrichton)
[@loganek](https://github.com/loganek)

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
[`wasi-libc`] meaning that a WebAssembly-targeting-Clang is not required to
use the WASI targets from Rust.  If there is no actual interoperation with C
then `rustup target add wasm32-wasip1` is all that's needed to get
started with WASI.

Note that this behavior can be controlled with `-Clinker` and
`-Clink-self-contained`, however. By specifying `clang` as a linker and
disabling the `link-self-contained` option an external version of `libc.a` can
be used instead.

[`wasi-libc`]: https://github.com/WebAssembly/wasi-libc

## Building the target

To build this target first acquire a copy of
[`wasi-sdk`](https://github.com/WebAssembly/wasi-sdk/). At this time version 22
is the minimum needed.

Next configure the `WASI_SDK_PATH` environment variable to point to where this
is installed. For example:

```text
export WASI_SDK_PATH=/path/to/wasi-sdk-22.0
```

Next be sure to enable LLD when building Rust from source as LLVM's `wasm-ld`
driver for LLD is required when linking WebAssembly code together. Rust's build
system will automatically pick up any necessary binaries and programs from
`WASI_SDK_PATH`.

## Building Rust programs

The `wasm32-wasip1` target is shipped with rustup so users can install
the target with:

```text
rustup target add wasm32-wasip1
```

Rust programs can be built for that target:

```text
rustc --target wasm32-wasip1 your-code.rs
```

## Cross-compilation

This target can be cross-compiled from any hosts.

## Testing

This target is tested in rust-lang/rust CI on all merges. A subset of tests are
run in the `test-various` builder such as the UI tests and libcore tests. This
can be tested locally, for example, with:

```text
./x.py test --target wasm32-wasip1 tests/ui
```

## Conditionally compiling code

It's recommended to conditionally compile code for this target with:

```text
#[cfg(all(target_os = "wasi", target_env = "p1"))]
```

Note that the `target_env = "p1"` condition first appeared in Rust 1.80. Prior
to Rust 1.80 the `target_env` condition was not set.

## Enabled WebAssembly features

The default set of WebAssembly features enabled for compilation is currently the
same as [`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md). See the
documentation there for more information.
