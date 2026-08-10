# `wasm32-wasip3`

**Tier: 2**

The `wasm32-wasip3` target is the next stage of evolution of the
[`wasm32-wasip2`](./wasm32-wasip2.md) target. The `wasm32-wasip3` target enables
the Rust standard library to use WASIp3 APIs to implement various pieces of
functionality. WASIp3 brings native async support over WASIp2, which integrates
well with Rust's `async` ecosystem. Additionally a future release of Rust's
`wasm32-wasip3` target will support cooperative threading and `std::thread`
APIs.

The original proposal for adding this target can be found in
[rust-lang/compiler-team#1001] and this target is first available on stable in
Rust 1.100.0. A notable major change from historical WebAssembly targets is that
the ABI of this target is slightly different. The linear memory shadow stack
pointer is stored in a component model task context slot instead of a
WebAssembly` global`. Additionally the base pointer of TLS is managed
differently than other targets. These changes are made to enable cooperative
multithreading on this target.

> **Note**: As of 2026-09-03 cooperative multithreading is not yet supported on
> this target in Rust. The component model specification and library support
> work for this is in-development and not yet complete, but it's expected to be
> complete before the end of the year. Before that time spawning a thread via
> `std::thread` will return an error. Note though that this support can be
> tested through the [instructions below](#testing-cooperative-multithreading).

[rust-lang/compiler-team#1001]: https://github.com/rust-lang/compiler-team/issues/1001

## Target maintainers

[@alexcrichton](https://github.com/alexcrichton)
[@yoshuawuyts](https://github.com/yoshuawuyts)

## Requirements

This target is cross-compiled. The target supports `std` fully. This target
requires LLVM 23 to be used and additionally requires `wasi-sdk-34`-or-later if
you're building it locally or linking with this externally.

## Platform requirements

WebAssembly runtimes that want to execute components compiled for this target
must support WASI 0.3.0 and the requisite required component model features
(notably async). Two example runtimes for WASIp3 are [Wasmtime] and [Jco].

[Wasmtime]: https://wasmtime.dev/
[Jco]: https://github.com/bytecodealliance/jco

## Building the target

To build this target first acquire a copy of
[`wasi-sdk`](https://github.com/WebAssembly/wasi-sdk/). At this time version 34
is the minimum needed.

Next configure the `WASI_SDK_PATH` environment variable to point to where this
is installed. For example:

```text
export WASI_SDK_PATH=/path/to/wasi-sdk-34.0
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

## Testing Cooperative Multithreading

The [component model specification][spec] is in the process of adding intrinsics
to support cooperative multithreading in a component guest. These intrinsics can
be found in the [explainer] and are all gated by the 🧵 emoji. Support for
cooperative multithreading is a work-in-progress and not yet complete, but the
adventurous can configure this target to go ahead and test things out.

The majority of changes necessary to get cooperative multithreading lie within
Rust's [wasi-libc dependency][wasi-libc]. This means that to test cooperative
multithreading a different build than the default wasi-libc needs to be used.
Starting with [wasi-sdk-34] there is a temporary sysroot which contains support
for a wasip3 target that has multithreading enabled in wasi-libc. To test out
the `wasm32-wasip3` Rust target with threads your compilation needs to be
configured to use this sysroot.

An example of doing this is this program:

```rust
fn main() {
    std::thread::spawn(|| {
        println!("hi");
    })
    .join()
    .unwrap();
}
```

is compiled and run by default as:

```console
$ rustc foo.rs --target wasm32-wasip3
$ wasmtime foo.wasm

thread 'main' (1) panicked at library/std/src/thread/functions.rs:131:29:
failed to spawn thread: Os { code: 58, kind: Unsupported, message: "Not supported" }
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
Error: failed to run main module `foo.wasm`

...
```

which shows that by default threads cannot be spawned. By configuring a custom
sysroot to be used, however:

```console
$ rustc foo.rs --target wasm32-wasip3 \
    -Clink-self-contained=n \
    -Clinker=$WASI_SDK_PATH/bin/wasm32-wasip3-clang \
    -Clink-arg=--sysroot=$WASI_SDK_PATH/share/wasi-sysroot/experimental-coop-threads \
    -Clink-arg=-Wl,--export=cabi_realloc
$ wasmtime -W component-model-threading foo.wasm
hi
```

Here `-Clink-self-contained=n` avoids the vendored files in the standard library
which come from a build of wasi-libc incompatible with cooperative
multithreading. The `-Clinker` flag changes to use `clang` to be able to pass a
custom `--sysroot` argument and follow its logic for startup objects. The
`--sysroot` flag then points to the experimental sysroot for coop threads and
`--export` is required right now as a minor workaround.

When compiling with Cargo you can use these environment variables:

```console
$ export CARGO_TARGET_WASM32_WASIP3_LINKER=$WASI_SDK_PATH/bin/wasm32-wasip3-clang
$ export CARGO_TARGET_WASM32_WASIP3_RUNNER='wasmtime -W component-model-threading'
$ export CARGO_TARGET_WASM32_WASIP3_RUSTFLAGS="\
    -Clink-self-contained=n \
    -Clink-arg=--sysroot=$WASI_SDK_PATH/share/wasi-sysroot/experimental-coop-threads \
    -Clink-arg=-Wl,--export=cabi_realloc"
$ cargo run --target wasm32-wasip3
hi
```

Standard synchronization primitives in `std::thread` and `std::sync` should all
work on this target with cooperative multithreading. Should you run into any
issues please don't hesitate to file an issue and cc the target maintainers.

Note that it is currently intended that by the end of 2026 this support will all
be enabled by default and this section of the documentation will be deleted
since working with threads should "just work".

[spec]: https://github.com/webassembly/component-model
[explainer]: https://github.com/WebAssembly/component-model/blob/main/design/mvp/Explainer.md
[wasi-libc]: https://github.com/webassembly/wasi-libc
[wasi-sdk-34]: https://github.com/WebAssembly/wasi-sdk/releases/tag/wasi-sdk-34
