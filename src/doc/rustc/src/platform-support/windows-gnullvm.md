# \*-windows-gnullvm

**Tier: 2 (without host tools)**

Windows targets similar to `*-windows-gnu` but using UCRT as the runtime and various LLVM tools/libraries instead of GCC/Binutils.

Target triples available so far:
- `aarch64-pc-windows-gnullvm`
- `i686-pc-windows-gnullvm`
- `x86_64-pc-windows-gnullvm`

## Target maintainers

- [@mati865](https://github.com/mati865)
- [@thomcc](https://github.com/thomcc)

## Requirements

The easiest way to obtain these targets is cross-compilation, but native build from `x86_64-pc-windows-gnu` is possible with few hacks which I don't recommend.
Std support is expected to be on par with `*-windows-gnu`.

Binaries for this target should be at least on par with `*-windows-gnu` in terms of requirements and functionality.

Those targets follow Windows calling convention for `extern "C"`.

Like with any other Windows target, created binaries are in PE format.

## Building the target

These targets can be easily cross-compiled
using [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) toolchain or [MSYS2 CLANG*](https://www.msys2.org/docs/environments/) environments.
Just fill `[target.*]` sections for both build and resulting compiler and set installation prefix in `bootstrap.toml`.
Then run `./x.py install`.
In my case I had ran `./x.py install --host x86_64-pc-windows-gnullvm --target x86_64-pc-windows-gnullvm` inside MSYS2 MINGW64 shell
so `x86_64-pc-windows-gnu` was my build toolchain.

Native bootstrapping is doable in two ways:
- cross-compile gnullvm host toolchain and use it as build toolchain for the next build,
- copy libunwind libraries and rename them to mimic libgcc like here: https://github.com/msys2/MINGW-packages/blob/68e640756df2df6df6afa60f025e3f936e7b977c/mingw-w64-rust/PKGBUILD#L108-L109, stage0 compiler will be mostly broken but good enough to build the next stage.

The second option might stop working anytime, so it's not recommended.

## Building Rust programs

Rust does ship a pre-compiled std library for those targets.
That means one can easily cross-compile for those targets from other hosts if C proper toolchain is installed.

Alternatively full toolchain can be built as described in the previous section.

## Testing

Created binaries work fine on Windows or Wine using native hardware. Testing AArch64 on x86_64 is problematic though and requires spending some time with QEMU.
Most of x86_64 testsuite does pass when cross-compiling,
with exception for `rustdoc` and `ui-fulldeps` that fail with and error regarding a missing library,
they do pass in native builds though.
The only failing test is std's `process::tests::test_proc_thread_attributes` for unknown reason.

## Cross-compilation toolchains and C code

Compatible C code can be built with Clang's `aarch64-pc-windows-gnu`, `i686-pc-windows-gnullvm` and `x86_64-pc-windows-gnu` targets as long as LLVM-based C toolchains are used.
Those include:
- [llvm-mingw](https://github.com/mstorsjo/llvm-mingw)
- [MSYS2 with CLANG* environment](https://www.msys2.org/docs/environments)
