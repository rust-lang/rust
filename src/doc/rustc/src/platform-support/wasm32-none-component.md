# `wasm32-none-component`

**Tier: 3**

The `wasm32-none-component` target is a `no_std` WebAssembly compilation target. This
target produces a WebAssembly Component as the output and otherwise makes no
assumptions about the environment that it is compiled within. As a `no_std`
compilation target you're required to bring your own `#[global_allocator]` and
`#[panic_handler`].

The purpose of this target is to represent a compilation target the produces a
WebAssembly component which otherwise does not bring in any APIs by default.
This is suitable for testing out new component targets, for example, or for
building possibly leaner components than one might acquire with the
`wasm32-wasip{2,3}` targets, for example.

Components produced for this target are not assumed to have any particular host
that they're going to run within, so ecosystem crates are expected to fall back
to `no_std` paths or wasm paths otherwise.

One example usage of this target is to build a custom SDK based on the
`wit-bindgen` crate on crates.io. This provides users the ability to import
functions into a component, but this is an opt-in feature per build and
otherwise won't happen by default. This can additionally be used to define
exports as well.

## Target maintainers

[@alexcrichton](https://github.com/alexcrichton)

## Requirements

This target is cross-compiled and has no requirements beyond the base LLVM
toolchain. The target uses `wasm-component-ld` to link which requires LLD to be
built from the Rust source tree, and this ships by default with compiler builds.

## Conditionally compiling code

It's recommended to conditionally compile code for this target with:

```text
#[cfg(all(target_family = "wasm", target_os = "none"))]
```

It's worth noting though that this does not specifically select this target
uniquely as this also matches the `wasm32v1-none` target, for example. Detecting
this target and this target alone currently requires a `build.rs` to look at the
`TARGET` being compiled for to look for the `wasm32-none-component` target name.

## Enabled WebAssembly features

The default set of WebAssembly features enabled for compilation is currently the
same as [`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md). See the
documentation there for more information.
