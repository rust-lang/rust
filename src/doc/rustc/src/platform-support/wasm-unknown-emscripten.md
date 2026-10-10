# `wasm*-unknown-emscripten`

Emscripten WebAssembly targets.

**Tier: 2 (without Host Tools)**

- `wasm32-unknown-emscripten`: WebAssembly via Emscripten.

**Tier: 3**

- `wasm64-unknown-emscripten`: WebAssembly with Memory64 via Emscripten.

These targets are WebAssembly compilation targets which use the
[Emscripten](https://emscripten.org/) compiler toolchain. Emscripten is primarily
a C/C++ toolchain designed to make it as easy as possible to port C/C++ code
written for Linux to run on the web or in other JavaScript runtimes such as Node.
It thus provides POSIX-compatible (musl) `libc` and `libstd` implementations and
many Linux APIs, access to the OpenGL and SDL APIs, and the ability to run arbitrary
JavaScript code, all based on web APIs using JS glue code. With the
`wasm*-unknown-emscripten` targets, Rust code can interoperate with Emscripten's
ecosystem, C/C++ and JS code, and web APIs.

One example user of these targets is the [`pyodide` project](https://pyodide.org/)
which provides a Python runtime in WebAssembly using Emscripten and compiles Python
extension modules written in Rust to the `wasm32-unknown-emscripten` target.

If you want to generate a standalone WebAssembly binary that does not require
access to the web APIs or the Rust standard library, the
[`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md) or
[`wasm64-unknown-unknown`](./wasm64-unknown-unknown.md) targets may be better
suited for you. Those targets however do not (easily) support interop with
C/C++ code.

Like Emscripten, the WASI targets [`wasm32-wasip1`](./wasm32-wasip1.md),
[`wasm32-wasip2`](./wasm32-wasip2.md), and
[`wasm32-wasip3`](./wasm32-wasip3.md), also provide access to the host
environment, support interop with C/C++ (and other languages), and support most
of the Rust standard library. While the WASI targets are portable across
different hosts (web and non-web), WASI has no standard way of accessing web
APIs, whereas Emscripten has the ability to run arbitrary JS from WASM and
access many web APIs.  If you are only targeting the web and need to access web
APIs, these targets may be preferable.

## Target maintainers

`wasm32-unknown-emscripten`:
[@hoodmane](https://github.com/hoodmane)
[@juntyr](https://github.com/juntyr)

`wasm64-unknown-emscripten`:
[@hoodmane](https://github.com/hoodmane)

## Requirements

These targets are cross-compiled. The Emscripten compiler toolchain `emcc` must be
installed to link WASM binaries for these targets. Emscripten 4.0.0 or newer is
required. You can install `emcc` using:

```sh
git clone https://github.com/emscripten-core/emsdk.git --depth 1
./emsdk/emsdk install latest
./emsdk/emsdk activate latest
source ./emsdk/emsdk_env.sh
```

Please refer to <https://emscripten.org/docs/getting_started/downloads.html> for
further details and instructions.

## Building the target

Building this target can be done by:

* Configure the `wasm32-unknown-emscripten` or `wasm64-unknown-emscripten` target
  to get built.
* Ensure the `WebAssembly` target backend is not disabled in LLVM.

These are all controlled through `bootstrap.toml` options. It should be possible
to build these targets on any platform. A minimal example configuration would be:

```toml
[llvm]
targets = "WebAssembly"

[build]
build-stage = 1
target = ["wasm32-unknown-emscripten", "wasm64-unknown-emscripten"]

[rust]
lldb = true
```

## Building Rust programs

The `wasm32-unknown-emscripten` target is tier 2 and has a prebuilt standard library
available, so using it can be done by adding it via rustup:

```sh
$ rustup target add wasm32-unknown-emscripten
```

and then compiling with the target:

```sh
$ rustc foo.rs --target wasm32-unknown-emscripten
$ file foo.wasm
```

The `wasm64-unknown-emscripten` target is tier 3, and you must compile the standard
library yourself, such as with `-Zbuild-std`:

```sh
$ cargo +nightly build -Zbuild-std --target wasm64-unknown-emscripten
```

## Cross-compilation

These targets can be cross-compiled from any host.

## Emscripten ABI Compatibility

The Emscripten compiler toolchain does not follow a semantic versioning scheme
that clearly indicates when breaking changes to the ABI can be made.
Additionally, Emscripten offers many different ABIs even for a single version of
Emscripten depending on the linker flags used, e.g. `-fwasm-exceptions` and
`-sWASM_BIGINT`. If the ABIs do not match, your code may exhibit undefined
behaviour.

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

These targets are not extensively tested in CI for the rust-lang/rust repository. It
can be tested locally, for example, with:

```sh
EMCC_CFLAGS="-sSTACK_SIZE=1MB -sMAXIMUM_MEMORY=2GB -sALLOW_MEMORY_GROWTH -Wno-limited-postlink-optimizations" ./x.py test --target wasm32-unknown-emscripten,wasm64-unknown-emscripten --skip src/tools/linkchecker --skip src/tools/html-checker
```

To run these tests, both `emcc` and `node` need to be in your `$PATH`. You can
install `node`, for example, using `nvm` by following the instructions at
<https://github.com/nvm-sh/nvm#install--update-script>.

If you need to test WebAssembly compatibility *in general*, it is recommended
to test the [`wasm32-wasip1`](./wasm32-wasip1.md) target instead.

## Conditionally compiling code

It's recommended to conditionally compile code for these targets with:

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
instructions over time. These targets' default set of supported WebAssembly
features will additionally change over time. These targets inherit the default
settings of LLVM which typically, but not necessarily, matches the default
settings of Emscripten as well. At link time, `emcc` configures the
linker to use Emscripten's settings.

Please refer to the [`wasm32-unknown-unknown`](./wasm32-unknown-unknown.md)
target's documentation on which WebAssembly features Rust enables by default, how
features can be disabled, and how Rust code can be conditionally compiled based on
which features are enabled.

`wasm64-unknown-emscripten` has a different set of default target features, see
[`wasm64-unknown-unknown`](./wasm64-unknown-unknown.md) for details on those.

Note that Rust code compiled for these targets currently enables
`-fwasm-exceptions` (legacy WASM exceptions) by default unless the Rust code is
compiled with `-Cpanic=abort`.

Please refer to the [Emscripten ABI compatibility](#emscripten-abi-compatibility)
section to ensure that the features that are enabled do not cause an ABI mismatch
between your Rust code, the pre-compiled Rust standard library, and other code compiled
for Emscripten.
