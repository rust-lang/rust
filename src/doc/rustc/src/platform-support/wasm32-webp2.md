# `wasm32-webp2`

**Tier: 3**

A target that compiles to a WebAssembly component which is intended to be run in
a web browser.

The `wasm32-webp2` target is, currently, an experimental effort towards having
an official Rust target which can be used to target web browsers and the web in
general. This target creates WebAssembly components in the same manner as the
`wasm32-wasip{2,3}` targets, for example, but the component generated imports
web APIs instead of WASI APIs. This target is an effort in conjunction with
colleagues from Firefox working on implementing (currently) experimental support
for components in a web browser. The long-term vision for this target is that
components created here can be directly loaded into a browser without any JS
glue code needed.

## Target maintainers

[@alexcrichton](https://github.com/alexcrichton)

## Requirements

This target requires the LLVM WebAssembly backend, the `wasm-ld` linker from
LLVM, and the wasi-sdk distribution of wasi-libc compiled for the `wasm32-webp2`
target as well. This target does not support `rustc` itself and must be
cross-compiled from some other host. This target supports the standard library,
but much of it is stubbed out to return errors such as spawning a thread,
opening a file, or creating a TCP socket. Basic functionality such as getting
randomness and getting the current time work, however. The default allocator for
this target is the implementation of `malloc` in wasi-libc itself.

This target depends on wasi-libc compiled for the `wasm32-webp2` target. This
means that it can interoperate with C code compiled for `wasm32-webp2` as well,
and additionally supports the POSIX-style interfaces that wasi-libc supports.

This target produces a WebAssembly component via the `wasm-component-ld` linker.

## Building the target

Building this Rust target requires adding `wasm32-webp2` to `bootstrap.toml` and
ensuring that the WebAssembly backend of LLVM is enabled. Once the target is
built, however, you'll need to additionally make sure you're able to acquire a
build of wasi-sdk with the `wasm32-webp2` target. As of the time of this
writing, 2026-11-13, wasi-sdk-35 is not yet released which will be the first
release with `wasm32-webp2` as a target. Until that happens you'll need to build
wasi-sdk from source.

Once the Rust target is built, you can produce a component with:

```console
$ ./build/$host/stage1/bin/rustc foo.rs \
    --target wasm32-webp2 \
    -C linker=$WASI_SDK_PATH/bin/clang
```

where `$WASI_SDK_PATH` points to wasi-sdk-35 or later. The final output
component can be inspected with `wasm-tools` for example to see that it imports
web and JS APIs:

```console
$ wasm-tools component wit foo.wasm
package root:component;

world root {
  import wasm:js/global;
  @external-id("wasm:js/global/console")
  import wasm:js/console;
}
...
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `std` by using
`build-std` or similar. When using `build-std` you'll need to ensure that you
have wasi-sdk-35 or later as mentioned above, and you'll additionally need to
configure `-Clinker` to point to the `clang` binary.

## Testing

There is currently no smooth path to testing these binaries at this time. The
binaries are intended to be loaded into a web browsers, and Firefox has not yet
shipped experimental support for components. In the future as this target
evolves it's expected that this testing story will be smoothed out over time.

## Cross-compilation toolchains and C code

This target supports interoperating with C that is compiled with wasi-sdk. The C
code must be compiled with the `wasm32-webp2` target.

## Conditionally compiling code

It's recommended to conditionally compile code for this target with:

```text
#[cfg(target_os = "web")]
```

or

```text
#[cfg(all(target_os = "web", target_env = "p2"))]
```

## Enabled WebAssembly features

The default set of WebAssembly features enabled for compilation is currently the
same as [`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md). See the
documentation there for more information.
