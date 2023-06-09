# \*-pc-windows-gnullvm

**Tier: 3**

Windows targets similar to `*-pc-windows-gnu` but using UCRT as the runtime and various LLVM tools/libraries instead of GCC/Binutils.

Target triples available so far:
- `aarch64-pc-windows-gnullvm`
- `x86_64-pc-windows-gnullvm`

## Target maintainers

- [@mati865](https://github.com/mati865)

## Requirements

The easiest way to obtain these targets is cross-compilation but native build from `x86_64-pc-windows-gnu` is possible with few hacks which I don't recommend.
Std support is expected to be on pair with `*-pc-windows-gnu`.

Binaries for this target should be at least on pair with `*-pc-windows-gnu` in terms of requirements and functionality.

Those targets follow Windows calling convention for `extern "C"`.

Like with any other Windows target created binaries are in PE format.

## Building the target

For cross-compilation I recommend using [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) toolchain, one change that seems necessary beside configuring cross compilers is disabling experimental `m86k` target. Otherwise LLVM build fails with `multiple definition ...` errors.
Native bootstrapping builds require rather fragile hacks until host artifacts are available so I won't describe them here.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Created binaries work fine on Windows or Wine using native hardware. Testing AArch64 on x86_64 is problematic though and requires spending some time with QEMU.
Once these targets bootstrap themselves on native hardware they should pass Rust testsuite.

## Cross-compilation toolchains and C code

Compatible C code can be built with Clang's `aarch64-pc-windows-gnu` and `x86_64-pc-windows-gnu` targets as long as LLVM based C toolchains are used.
Those include:
- [llvm-mingw](https://github.com/mstorsjo/llvm-mingw)
- [MSYS2 with CLANG* environment](https://www.msys2.org/docs/environments)
