# `wasm32v1-none`

**Tier: 2**

The `wasm32v1-none` target is a WebAssembly compilation target that:

- Imports nothing from its host environment
- Enables no proposals / features past the [W3C WebAssembly Core 1.0 spec]

[W3C WebAssembly Core 1.0 spec]: https://www.w3.org/TR/wasm-core-1/

The target is very similar to [`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md) and similarly uses LLVM's `wasm32-unknown-unknown` backend target. It contains only three minor differences:

* Setting the `target-cpu` to `mvp` rather than the default `generic`. Requesting `mvp` disables _all_ WebAssembly proposals / LLVM target feature flags.
* Enabling the [Import/Export of Mutable Globals] proposal (i.e. the `+mutable-globals` LLVM target feature flag)
* Not compiling the `std` library at all, rather than compiling it with stubs.

[Import/Export of Mutable Globals]: https://github.com/WebAssembly/mutable-global

## Target maintainers

[@alexcrichton](https://github.com/alexcrichton)
[@graydon](https://github.com/graydon)

## Requirements

This target is cross-compiled. It does not support `std`, only `core` and `alloc`. Since it imports nothing from its environment, any `std` parts that use OS facilities would be stubbed out with functions-that-fail anyways, and the experience of working with the stub `std` in the `wasm32-unknown-unknown` target was deemed not something worth repeating here.

Everything else about this target's requirements, building, usage and testing is the same as what's described in the [`wasm32-unknown-unknown` document](./wasm32-unknown-unknown.md), just using the target string `wasm32v1-none` in place of `wasm32-unknown-unknown`.

## Conditionally compiling code

It's recommended to conditionally compile code for this target with:

```text
#[cfg(all(target_family = "wasm", target_os = "none"))]
```

Note that there is no way to tell via `#[cfg]` whether code will be running on
the web or not.

## Enabled WebAssembly features

As noted above, _no WebAssembly proposals past 1.0_ are enabled on this target by default. Indeed, the entire point of this target is to have a way to compile for a stable "no post-1.0 proposals" subset of WebAssembly _on stable Rust_.

The [W3C WebAssembly Core 1.0 spec] was adopted as a W3C recommendation in December 2019, and includes exactly one "post-MVP" proposal: the [Import/Export of Mutable Globals] proposal.

All subsequent proposals are _disabled_ on this target by default, though they can be individually enabled by passing LLVM target-feature flags.

For reference sake, the set of proposals that LLVM supports at the time of writing, that this target _does not enable by default_, are listed here along with their LLVM target-feature flags:

* Post-1.0 proposals (integrated into the WebAssembly core 2.0 spec):
    * [Bulk memory] - `+bulk-memory`
    * [Sign-extending operations] - `+sign-ext`
    * [Non-trapping fp-to-int operations] - `+nontrapping-fptoint`
    * [Multi-value] - `+multivalue`
    * [Reference Types] - `+reference-types`
    * [Fixed-width SIMD] - `+simd128`
* Post-2.0 proposals:
    * [Threads] (supported by atomics) - `+atomics`
    * [Exception handling]  - `+exception-handling`
    * [Extended Constant Expressions]  - `+extended-const`
    * [Half Precision]  - `+half-precision`
    * [Multiple memories]- `+multimemory`
    * [Relaxed SIMD] - `+relaxed-simd`
    * [Tail call] - `+tail-call`

[Bulk memory]: https://github.com/WebAssembly/spec/blob/main/proposals/bulk-memory-operations/Overview.md
[Sign-extending operations]: https://github.com/WebAssembly/spec/blob/main/proposals/sign-extension-ops/Overview.md
[Non-trapping fp-to-int operations]: https://github.com/WebAssembly/spec/blob/main/proposals/nontrapping-float-to-int-conversion/Overview.md
[Multi-value]: https://github.com/WebAssembly/spec/blob/main/proposals/multi-value/Overview.md
[Reference Types]: https://github.com/WebAssembly/spec/blob/main/proposals/reference-types/Overview.md
[Fixed-width SIMD]: https://github.com/WebAssembly/spec/blob/main/proposals/simd/SIMD.md
[Threads]: https://github.com/webassembly/threads
[Exception handling]: https://github.com/WebAssembly/exception-handling
[Extended Constant Expressions]: https://github.com/WebAssembly/extended-const
[Half Precision]: https://github.com/WebAssembly/half-precision
[Multiple memories]: https://github.com/WebAssembly/multi-memory
[Relaxed SIMD]: https://github.com/WebAssembly/relaxed-simd
[Tail call]: https://github.com/WebAssembly/tail-call

Additional proposals in the future are, of course, also not enabled by default.

## Rationale relative to wasm32-unknown-unknown

As noted in the [`wasm32-unknown-unknown` document](./wasm32-unknown-unknown.md), it is possible to compile with `--target wasm32-unknown-unknown` and disable all WebAssembly proposals "by hand", by passing `-Ctarget-cpu=mvp`. Furthermore one can enable proposals one by one by passing LLVM target feature flags, such as `-Ctarget-feature=+mutable-globals`.

Is it therefore reasonable to wonder what the difference is between building with this:

```sh
$ rustc --target wasm32-unknown-unknown -Ctarget-cpu=mvp -Ctarget-feature=+mutable-globals
```

and building with this:

```sh
$ rustc --target wasm32v1-none
```

The difference is in how the `core` and `alloc` crates are compiled for distribution with the toolchain, and whether it works on _stable_ Rust toolchains or requires _nightly_ ones. Again referring back to the [`wasm32-unknown-unknown` document](./wasm32-unknown-unknown.md), note that to disable all post-MVP proposals on that target one _actually_ has to compile with this:

```sh
$ export RUSTFLAGS="-Ctarget-cpu=mvp -Ctarget-feature=+mutable-globals"
$ cargo +nightly build -Zbuild-std=panic_abort,std --target wasm32-unknown-unknown
```

Which not only rebuilds `std`, `core` and `alloc` (which is somewhat costly and annoying) but more importantly requires the use of nightly Rust toolchains (for the `-Zbuild-std` flag). This is very undesirable for the target audience, which consists of people targeting WebAssembly implementations that prioritize stability, simplicity and/or security over feature support.

This `wasm32v1-none` target exists as an alternative option that works on stable Rust toolchains, without rebuilding the stdlib.
