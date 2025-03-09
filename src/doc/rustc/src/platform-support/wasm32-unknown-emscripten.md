# `wasm32-unknown-emscripten`

**Tier: 2**

The `wasm32-unknown-emscripten` target is a WebAssembly compilation target which
uses the [Emscripten](https://emscripten.org/) compiler toolchain. Emscripten is
a C/C++ toolchain designed to make it as easy as possible to port C/C++ code
written for Linux to run on the web or in other JavaScript runtimes such as Node.
It thus provides POSIX-compatible (musl) `libc` and `libstd` implementations and
many Linux APIs, access to the OpenGL and SDL APIs, and the ability to run arbitrary
JavaScript code, all based on web APIs using JS glue code. With the
`wasm32-unknown-emscripten` target, Rust code can interoperate with Emscripten's
ecosystem, C/C++ and JS code, and web APIs.

One existing user of this target is the
[`pyodide` project](https://pyodide.org/) which provides a Python runtime in
WebAssembly using Emscripten and compiles Python extension modules written in Rust
to the `wasm32-unknown-emscripten` target.

If you want to generate a standalone WebAssembly binary that does not require
access to the web APIs or the Rust standard library, the
[`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md) target may be better
suited for you. However, [`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md)
does not (easily) support interop with C/C++ code. Please refer to the
[wasm-bindgen](https://crates.io/crates/wasm-bindgen) crate in case you want to
interoperate with JavaScript with this target.

Like Emscripten, the WASI targets [`wasm32-wasip1`](./wasm32-wasip1.md) and
[`wasm32-wasip2`](./wasm32-wasip2.md) also provide access to the host environment,
support interop with C/C++ (and other languages), and support most of the Rust
standard library. While the WASI targets are portable across different hosts
(web and non-web), WASI has no standard way of accessing web APIs, whereas
Emscripten has the ability to run arbitrary JS from WASM and access many web APIs.
If you are only targeting the web and need to access web APIs, the
`wasm32-unknown-emscripten` target may be preferable.

## Target maintainers

- Hood Chatham, https://github.com/hoodmane
- Juniper Tyree, https://github.com/juntyr

## Requirements

This target is cross-compiled. The Emscripten compiler toolchain `emcc` must be
installed to link WASM binaries for this target. You can install `emcc` using:

```sh
git clone https://github.com/emscripten-core/emsdk.git --depth 1
./emsdk/emsdk install 3.1.68
./emsdk/emsdk activate 3.1.68
source ./emsdk/emsdk_env.sh
```

Please refer to <https://emscripten.org/docs/getting_started/downloads.html> for
further details and instructions.

## Building the target

Building this target can be done by:

* Configure the `wasm32-unknown-emscripten` target to get built.
* Ensure the `WebAssembly` target backend is not disabled in LLVM.

These are all controlled through `bootstrap.toml` options. It should be possible
to build this target on any platform. A minimal example configuration would be:

```toml
[llvm]
targets = "WebAssembly"

[build]
build-stage = 1
target = ["wasm32-unknown-emscripten"]
```

## Building Rust programs

Rust programs can be compiled by adding this target via rustup:

```sh
$ rustup target add wasm32-unknown-emscripten
```

and then compiling with the target:

```sh
$ rustc foo.rs --target wasm32-unknown-emscripten
$ file foo.wasm
```

## Cross-compilation

This target can be cross-compiled from any host.

## Emscripten ABI Compatibility

The Emscripten compiler toolchain does not follow a semantic versioning scheme
that clearly indicates when breaking changes to the ABI can be made. Additionally,
Emscripten offers many different ABIs even for a single version of Emscripten
depending on the linker flags used, e.g. `-fexceptions` and `-sWASM_BIGINT`. If
the ABIs mismatch, your code may exhibit undefined behaviour.

To ensure that the ABIs of your Rust code, of the Rust standard library, and of
other code compiled for Emscripten all match, you should rebuild the Rust standard
library with your local Emscripten version and settings using:

```sh
cargo +nightly -Zbuild-std build
```

If you still want to use the pre-compiled `std` from rustup, you should ensure
that your local Emscripten matches the version used by Rust and be careful about
any `-C link-arg`s that you compiled your Rust code with.

## Testing

This target is not extensively tested in CI for the rust-lang/rust repository. It
can be tested locally, for example, with:

```sh
EMCC_CFLAGS="-s MAXIMUM_MEMORY=2GB" ./x.py test --target wasm32-unknown-emscripten --skip src/tools/linkchecker
```

To run these tests, both `emcc` and `node` need to be in your `$PATH`. You can
install `node`, for example, using `nvm` by following the instructions at
<https://github.com/nvm-sh/nvm#install--update-script>.

If you need to test WebAssembly compatibility *in general*, it is recommended
to test the [`wasm32-wasip1`](./wasm32-wasip1.md) target instead.

## Conditionally compiling code

It's recommended to conditionally compile code for this target with:

```text
#[cfg(target_os = "emscripten")]
```

It may sometimes be necessary to conditionally compile code for WASM targets
which do *not* use emscripten, which can be achieved with:

```text
#[cfg(all(target_family = "wasm", not(target_os = "emscripten)))]
```

## Enabled WebAssembly features

WebAssembly is an evolving standard which adds new features such as new
instructions over time. This target's default set of supported WebAssembly
features will additionally change over time. The `wasm32-unknown-emscripten` target
inherits the default settings of LLVM which typically, but not necessarily, matches
the default settings of Emscripten as well. At link time, `emcc` configures the
linker to use Emscripten's settings.

Please refer to the [`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md)
target's documentation on which WebAssembly features Rust enables by default, how
features can be disabled, and how Rust code can be conditionally compiled based on
which features are enabled.

Note that Rust code compiled for `wasm32-unknown-emscripten` currently enables
`-fexceptions` (JS exceptions) by default unless the Rust code is compiled with
`-Cpanic=abort`. `-fwasm-exceptions` (WASM exceptions) is not yet currently supported,
see <https://github.com/rust-lang/rust/issues/112195>.

Please refer to the [Emscripten ABI compatibility](#emscripten-abi-compatibility)
section to ensure that the features that are enabled do not cause an ABI mismatch
between your Rust code, the pre-compiled Rust standard library, and other code compiled
for Emscripten.
