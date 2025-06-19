# `wasm32-unknown-unknown`

**Tier: 2**

The `wasm32-unknown-unknown` target is a WebAssembly compilation target which
does not import any functions from the host for the standard library. This is
the "minimal" WebAssembly in the sense of making the fewest assumptions about
the host environment. This target is often used when compiling to the web or
JavaScript environments as there is no standard for what functions can be
imported on the web. This target can also be useful for creating minimal or
bare-bones WebAssembly binaries.

The `wasm32-unknown-unknown` target has support for the Rust standard library
but many parts of the standard library do not work and return errors. For
example `println!` does nothing, `std::fs` always return errors, and
`std::thread::spawn` will panic. There is no means by which this can be
overridden. For a WebAssembly target that more fully supports the standard
library see the [`wasm32-wasip1`](./wasm32-wasip1.md) or
[`wasm32-wasip2`](./wasm32-wasip2.md) targets.

The `wasm32-unknown-unknown` target has full support for the `core` and `alloc`
crates. It additionally supports the `HashMap` type in the `std` crate, although
hash maps are not randomized like they are on other platforms.

One existing user of this target (please feel free to edit and expand this list
too) is the [`wasm-bindgen` project](https://github.com/rustwasm/wasm-bindgen)
which facilitates Rust code interoperating with JavaScript code. Note, though,
that not all uses of `wasm32-unknown-unknown` are using JavaScript and the web.

## Target maintainers

When this target was added to the compiler, platform-specific documentation here
was not maintained at that time. This means that the list below is not
exhaustive, and there are more interested parties in this target. That being
said, those interested in maintaining this target are:

[@alexcrichton](https://github.com/alexcrichton)

## Requirements

This target is cross-compiled. The target includes support for `std` itself,
but as mentioned above many pieces of functionality that require an operating
system do not work and will return errors.

This target currently has no equivalent in C/C++. There is no C/C++ toolchain
for this target. While interop is theoretically possible it's recommended to
instead use one of:

* [`wasm32-unknown-emscripten`](./wasm32-unknown-emscripten.md) - for web-based
  use cases the Emscripten toolchain is typically chosen for running C/C++.
* [`wasm32-wasip1`](./wasm32-wasip1.md) - the wasi-sdk toolchain is used to
  compile C/C++ on this target and can interop with Rust code. WASI works on
  the web so far as there's no blocker, but an implementation of WASI APIs
  must be either chosen or reimplemented.

This target has no build requirements beyond what's in-tree in the Rust
repository. Linking binaries requires LLD to be enabled for the `wasm-ld`
driver. This target uses the `dlmalloc` crate as the default global allocator.

## Building the target

Building this target can be done by:

* Configure the `wasm32-unknown-unknown` target to get built.
* Configure LLD to be built.
* Ensure the `WebAssembly` target backend is not disabled in LLVM.

These are all controlled through `bootstrap.toml` options. It should be possible
to build this target on any platform.

## Building Rust programs

Rust programs can be compiled by adding this target via rustup:

```sh
$ rustup target add wasm32-unknown-unknown
```

and then compiling with the target:

```sh
$ rustc foo.rs --target wasm32-unknown-unknown
$ file foo.wasm
```

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

This target is not tested in CI for the rust-lang/rust repository. Many tests
must be disabled to run on this target and failures are non-obvious because
`println!` doesn't work in the standard library. It's recommended to test the
`wasm32-wasip1` target instead for WebAssembly compatibility.

## Conditionally compiling code

It's recommended to conditionally compile code for this target with:

```text
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
```

Note that there is no way to tell via `#[cfg]` whether code will be running on
the web or not.

## Enabled WebAssembly features

WebAssembly is an evolving standard which adds new features such as new
instructions over time. This target's default set of supported WebAssembly
features will additionally change over time. The `wasm32-unknown-unknown` target
inherits the default settings of LLVM which typically matches the default
settings of Emscripten as well.

Changes to WebAssembly go through a [proposals process][proposals] but reaching
the final stage (stage 5) does not automatically mean that the feature will be
enabled in LLVM and Rust by default. At this time the general guidance is that
features must be present in most engines for a "good chunk of time" before
they're enabled in LLVM by default. There is currently no exact number of
months or engines that are required to enable features by default.

[proposals]: https://github.com/WebAssembly/proposals

As of the time of this writing the proposals that are enabled by default (the
`generic` CPU in LLVM terminology) are:

* `multivalue`
* `mutable-globals`
* `reference-types`
* `sign-ext`
* `nontrapping-fptoint` (Rust 1.87.0+, LLVM 20+)
* `bulk-memory` (Rust 1.87.0+, LLVM 20+)

If you're compiling WebAssembly code for an engine that does not support a
feature in LLVM's default feature set then the feature must be disabled at
compile time. There are two approaches to choose from:

  - If you are targeting a feature set no smaller than the W3C WebAssembly Core
    1.0 recommendation -- which is equivalent to the WebAssembly MVP plus the
    `mutable-globals` feature -- and you are building `no_std`, then you can
    simply use the [`wasm32v1-none` target](./wasm32v1-none.md) instead of
    `wasm32-unknown-unknown`, which uses only those minimal features and
    includes a core and alloc library built with only those minimal features.

  - Otherwise -- if you need std, or if you need to target the ultra-minimal
    "MVP" feature set, excluding `mutable-globals` -- you will need to manually
    specify `-Ctarget-cpu=mvp` and also rebuild the stdlib using that target to
    ensure no features are used in the stdlib. This in turn requires use of a
    nightly compiler.

Compiling all code for the initial release of WebAssembly looks like:

```sh
$ export RUSTFLAGS=-Ctarget-cpu=mvp
$ cargo +nightly build -Zbuild-std=panic_abort,std --target wasm32-unknown-unknown
```

Here the `mvp` "cpu" is a placeholder in LLVM for disabling all supported
features by default. Cargo's `-Zbuild-std` feature, a Nightly Rust feature, is
then used to recompile the standard library in addition to your own code. This
will produce a binary that uses only the original WebAssembly features by
default and no proposals since its inception.

To enable individual features on either this target or `wasm32v1-none`, pass
arguments of the form `-Ctarget-feature=+foo`.  Available features for Rust code
itself are documented in the [reference] and can also be found through:

```sh
$ rustc -Ctarget-feature=help --target wasm32-unknown-unknown
```

You'll need to consult your WebAssembly engine's documentation to learn more
about the supported WebAssembly features the engine has.

[reference]: https://doc.rust-lang.org/reference/attributes/codegen.html#wasm32-or-wasm64

Note that it is still possible for Rust crates and libraries to enable
WebAssembly features on a per-function level. This means that the build
command above may not be sufficient to disable all WebAssembly features. If the
final binary still has SIMD instructions, for example, the function in question
will need to be found and the crate in question will likely contain something
like:

```rust,ignore (not-always-compiled-to-wasm)
#[target_feature(enable = "simd128")]
fn foo() {
    // ...
}
```

In this situation there is no compiler flag to disable emission of SIMD
instructions and the crate must instead be modified to not include this function
at compile time either by default or through a Cargo feature. For crate authors
it's recommended to avoid `#[target_feature(enable = "...")]` except where
necessary and instead use:

```rust,ignore (not-always-compiled-to-wasm)
#[cfg(target_feature = "simd128")]
fn foo() {
    // ...
}
```

That is to say instead of enabling target features it's recommended to
conditionally compile code instead. This is notably different to the way native
platforms such as x86\_64 work, and this is due to the fact that WebAssembly
binaries must only contain code the engine understands. Native binaries work so
long as the CPU doesn't execute unknown code dynamically at runtime.
