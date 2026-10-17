# `wasm-component-ld`

This wrapper is a wrapper around the [`wasm-component-ld`] crates.io crate.
That crate is itself a thin wrapper around two pieces:

* `wasm-ld` - the LLVM-based linker distributed as part of LLD and packaged in
  Rust as `rust-lld`.
* [`wit-component`] - a Rust crate for creating a [WebAssembly Component] from a
  core wasm module.

This linker is used for Rust's `wasm32-wasip2` target to natively output a
component instead of a core WebAssembly module, unlike other WebAssembly
targets. If you're confused about any of this here's an FAQ-style explanation of
what's going on here:

* **What's a component?** - It's a proposal to the WebAssembly standard
  primarily developed at this time by out-of-browser use cases of WebAssembly.
  You can find high-level documentation [here][component docs].

* **What's WASIp2?** - Not to be confused with WASIp1, WASIp0,
  `wasi_snapshot_preview1`, or `wasi_unstable`, it's a version of WASI. Released
  in January 2024 it's the first version of WASI defined in terms of the
  component model.

* **Why does this need its own linker?** - like any target that Rust has the
  `wasm32-wasip2` target needs a linker. What makes this different from other
  WebAssembly targets is that WASIp2 is defined at the component level, not core
  WebAssembly level. This means that filesystem functions take a `string`
  instead of `i32 i32`, for example. This means that the raw output of LLVM and
  `wasm-ld`, a core WebAssembly module, is not suitable.

* **Isn't writing a linker really hard?** - Generally, yes, but this linker
  works by first asking `wasm-ld` to do all the hard work. It invokes `wasm-ld`
  and then uses the output core WebAssembly module to create a component.

* **How do you create a component from a core module?** - this is the purpose of
  the [`wit-component`] crate, notably the `ComponentEncoder` type. This uses
  component type information embedded in the core module and a general set of
  conventions/guidelines with what the core module imports/exports. A component
  is then hooked up to codify all of these conventions in a component itself.

* **Why not require users to run `wit-component` themselves?** - while possible
  it adds friction to the usage `wasm32-wasip2` target. More importantly though
  the "module only" output of the `wasm32-wasip2` target is not ready right now.
  The standard library still imports from `wasi_snapshot_preview1` and it will
  take time to migrate all usage to WASIp2.

* **What exactly does this linker do?** - the `wasm-component-ld` has the same
  CLI interface and flags as `wasm-ld`, plus some more that are
  component-specific. These flags are used to forward most flags to `wasm-ld` to
  produce a core wasm module. After the core wasm module is produced the
  `wit-component` crate will read custom sections in the final binary which
  contain component type information. After merging all this type information
  together a component is produced which wraps the core module.

If you've got any other questions about this linker or its operation don't
hesitate to reach out to the maintainers of the `wasm32-wasip2` target.

[`wasm-component-ld`]: https://crates.io/crates/wasm-component-ld
[`wit-component`]: https://crates.io/crates/wit-component
[WebAssembly Component]: https://github.com/webassembly/component-model
[component docs]: https://component-model.bytecodealliance.org/
