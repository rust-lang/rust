# `wasm32-wasip2`

**Tier: 2**

The `wasm32-wasip2` target is a new and still (as of January 2024) an
experimental target. This target is an extension to `wasm32-wasip1` target,
originally known as `wasm32-wasi`. It is the next evolution in the development of
wasi (the [WebAssembly System Interface](https://wasi.dev)) that uses the WebAssembly
[component model] to allow for a standardized set of syscalls that are intended to empower
WebAssembly binaries with native host capabilities.

[component model]: https://github.com/WebAssembly/component-model

## Target maintainers

[@alexcrichton](https://github.com/alexcrichton)
[@rylev](https://github.com/rylev)

## Requirements

This target is cross-compiled. The target supports `std` fully.

## Platform requirements

The WebAssembly runtime should support the wasi preview 2 API set. Runtimes also
are required to support components since this target outputs a component as
opposed to a core wasm module. As of the time of this writing Wasmtime 17 and
above is able to run this target natively with no extra flags.

## Building the target in rustc

See the documentation for the [building the `wasm32-wasip1` target in
rustc](./wasm32-wasip1.md#building-the-target-in-rustc) for more information. The tl;dr;
is that [`wasi-sdk`] is required, and the `wasm32-wasip1` target documents the
minimum version required.

[`wasi-sdk`]: https://github.com/WebAssembly/wasi-sdk

## Building Rust programs

For more information see the documentation [`wasm32-wasip1`
target](./wasm32-wasip1.md#building-rust-programs). Replace `wasm32-wasip1`
target strings with `wasm32-wasip2`, however.

## Testing

This target is not tested in CI at this time. Locally it can be tested with a
`wasmtime` binary in `PATH` like so:

```text
./x.py test --target wasm32-wasip2 tests/ui
```

## Conditionally compiling code

It's recommended to conditionally compile code for this target with:

```text
#[cfg(all(target_os = "wasi", target_env = "p2"))]
```

## Enabled WebAssembly features

The default set of WebAssembly features enabled for compilation is currently the
same as [`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md). See the
documentation there for more information.

## Unwinding

This target is compiled with `-Cpanic=abort` by default. For information on
using `-Cpanic=unwind` see the [documentation about unwinding for
`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md#unwinding).
