# `wasm32-wasip2`

**Tier: 3**

The `wasm32-wasip2` target is a new and still (as of January 2024) an
experimental target. This target is an extension to `wasm32-wasip1` target,
originally known as `wasm32-wasi`. It is the next evolution in the development of
wasi (the [WebAssembly System Interface](https://wasi.dev)) that uses the WebAssembly
[component model] to allow for a standardized set of syscalls that are intended to empower
WebAssembly binaries with native host capabilities.

[component model]: https://github.com/WebAssembly/component-model

## Target maintainers

- Alex Crichton, https://github.com/alexcrichton
- Ryan Levick, https://github.com/rylev

## Requirements

This target is cross-compiled. The target supports `std` fully.

## Platform requirements

The WebAssembly runtime should support the wasi preview 2 API set.

This target is not a stable target. This means that there are only a few engines
which implement wasi preview 2, for example:

* Wasmtime - `-W component-model`
