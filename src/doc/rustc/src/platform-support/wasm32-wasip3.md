# `wasm32-wasip3`

**Tier: 3**

The `wasm32-wasip3` target is the next stage of evolution of the
[`wasm32-wasip2`](./wasm32-wasip2.md) target. The `wasm32-wasip3` target enables
the Rust standard library to use WASIp3 APIs to implement various pieces of
functionality. WASIp3 brings native async support over WASIp2, which integrates
well with Rust's `async` ecosystem.

> **Note**: As of 2025-10-01 WASIp3 has not yet been approved by the WASI
> subgroup of the WebAssembly Community Group. Development is expected to
> conclude in late 2025 or early 2026. Until then the Rust standard library
> won't actually use WASIp3 APIs on the `wasm32-wasip3` target as they are not
> yet stable and would reduce the stability of this target. Once WASIp3 is
> approved, however, the standard library will update to use WASIp3 natively.

> **Note**: This target does not yet build as of 2025-10-01 due to and update
> needed in the `libc` crate. Using it will require a `[patch]` for now.

> **Note**: Until the standard library is fully migrated to use the `wasip3`
> crate then components produced for `wasm32-wasip3` may import WASIp2 APIs.
> This is considered a transitionary phase until fully support of libstd is
> implemented.

## Target maintainers

[@alexcrichton](https://github.com/alexcrichton)

## Requirements

This target is cross-compiled. The target supports `std` fully.

## Platform requirements

The WebAssembly runtime should support both WASIp2 and WASIp3. Runtimes also
are required to support components since this target outputs a component as
opposed to a core wasm module. Two example runtimes for WASIp3 are [Wasmtime]
and [Jco].

[Wasmtime]: https://wasmtime.dev/
[Jco]: https://github.com/bytecodealliance/jco

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

## Testing

This target is not tested in CI at this time. Locally it can be tested with a
`wasmtime` binary in `PATH` like so:

```text
./x.py test --target wasm32-wasip3 tests/ui
```

## Conditionally compiling code

It's recommended to conditionally compile code for this target with:

```text
#[cfg(all(target_os = "wasi", target_env = "p3"))]
```

## Enabled WebAssembly features

The default set of WebAssembly features enabled for compilation is currently the
same as [`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md). See the
documentation there for more information.
