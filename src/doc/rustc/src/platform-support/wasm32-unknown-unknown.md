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

- Alex Crichton, https://github.com/alexcrichton

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

## Broken `extern "C"` ABI

This target has what is considered a broken `extern "C"` ABI implementation at
this time. Notably the same signature in Rust and C will compile to different
WebAssembly functions and be incompatible. This is considered a bug and it will
be fixed in a future version of Rust.

For example this Rust code:

```rust,ignore (does-not-link)
#[repr(C)]
struct MyPair {
    a: u32,
    b: u32,
}

extern "C" {
    fn take_my_pair(pair: MyPair) -> u32;
}

#[no_mangle]
pub unsafe extern "C" fn call_c() -> u32 {
    take_my_pair(MyPair { a: 1, b: 2 })
}
```

compiles to a WebAssembly module that looks like:

```wasm
(module
  (import "env" "take_my_pair" (func $take_my_pair (param i32 i32) (result i32)))
  (func $call_c
    i32.const 1
    i32.const 2
    call $take_my_pair
  )
)
```

The function when defined in C, however, looks like

```c
struct my_pair {
    unsigned a;
    unsigned b;
};

unsigned take_my_pair(struct my_pair pair) {
    return pair.a + pair.b;
}
```

```wasm
(module
  (import "env" "__linear_memory" (memory 0))
  (func $take_my_pair (param i32) (result i32)
    local.get 0
    i32.load offset=4
    local.get 0
    i32.load
    i32.add
  )
)
```

Notice how Rust thinks `take_my_pair` takes two `i32` parameters but C thinks it
only takes one.

The correct definition of the `extern "C"` ABI for WebAssembly is located in the
[WebAssembly/tool-conventions](https://github.com/WebAssembly/tool-conventions/blob/main/BasicCABI.md)
repository. The `wasm32-unknown-unknown` target (and only this target, not other
WebAssembly targets Rust support) does not correctly follow this document.

Example issues in the Rust repository about this bug are:

* [#115666](https://github.com/rust-lang/rust/issues/115666)
* [#129486](https://github.com/rust-lang/rust/issues/129486)

This current state of the `wasm32-unknown-unknown` backend is due to an
unfortunate accident which got relied on. The `wasm-bindgen` project prior to
0.2.89 was incompatible with the "correct" definition of `extern "C"` and it was
seen as not worth the tradeoff of breaking `wasm-bindgen` historically to fix
this issue in the compiler.

Thanks to the heroic efforts of many involved in this, however, `wasm-bindgen`
0.2.89 and later are compatible with the correct definition of `extern "C"` and
the nightly compiler currently supports a `-Zwasm-c-abi` implemented in
[#117919](https://github.com/rust-lang/rust/pull/117919). This nightly-only flag
can be used to indicate whether the spec-defined version of `extern "C"` should
be used instead of the "legacy" version of
whatever-the-Rust-target-originally-implemented. For example using the above
code you can see (lightly edited for clarity):

```shell
$ rustc +nightly -Zwasm-c-abi=spec foo.rs --target wasm32-unknown-unknown --crate-type lib --emit obj -O
$ wasm-tools print foo.o
(module
  (import "env" "take_my_pair" (func $take_my_pair (param i32) (result i32)))
  (func $call_c (result i32)
    ;; ...
  )
  ;; ...
)
```

which shows that the C and Rust definitions of the same function now agree like
they should.

The `-Zwasm-c-abi` compiler flag is tracked in
[#122532](https://github.com/rust-lang/rust/issues/122532) and a lint was
implemented in [#117918](https://github.com/rust-lang/rust/issues/117918) to
help warn users about the transition if they're using `wasm-bindgen` 0.2.88 or
prior. The current plan is to, in the future, switch `-Zwasm-c-api=spec` to
being the default. Some time after that the `-Zwasm-c-abi` flag and the
"legacy" implementation will all be removed. During this process users on a
sufficiently updated version of `wasm-bindgen` should not experience breakage.
