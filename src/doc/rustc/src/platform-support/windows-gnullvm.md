# \*-windows-gnullvm

**Tier: 2 (with host tools)**

Windows targets similar to `*-windows-gnu` but using UCRT as the runtime and various LLVM tools/libraries instead of
GCC/Binutils.

Target triples available so far:

- `aarch64-pc-windows-gnullvm`
- `i686-pc-windows-gnullvm`
- `x86_64-pc-windows-gnullvm`

## Target maintainers

[@mati865](https://github.com/mati865)
[@thomcc](https://github.com/thomcc)

## Requirements

Building those targets requires an LLVM-based C toolchain, for example, [llvm-mingw][1] or [MSYS2][2] with CLANG*
environment.

Binaries for this target should be at least on par with `*-windows-gnu` in terms of requirements and functionality,
except for implicit self-contained mode (explained in [the section below](#building-rust-programs)).

Those targets follow Windows calling convention for `extern "C"`.

Like with any other Windows target, created binaries are in PE format.

## Building the target

Both native and cross-compilation builds are supported and function similarly to other Rust targets.

## Building Rust programs

Rust ships both std and host tools for those targets. That allows using them as both the host and the target.

When used as the host and building pure Rust programs, no additional C toolchain is required.
The only requirements are to install `rust-mingw` component and to set `rust-lld` as the linker.
Otherwise, you will need to install the C toolchain mentioned previously.
There is no automatic fallback to `rust-lld` when the C toolchain is missing yet, but it may be added in the future.

## Testing

Created binaries work fine on Windows and Linux with Wine using native hardware.
Testing AArch64 on x86_64 is problematic, though, and requires launching a whole AArch64 system with QEMU.

Most of the x86_64 testsuite does pass, but because it isn't run on CI, different failures are expected over time.

## Cross-compilation toolchains and C code

Compatible C code can be built with Clang's `aarch64-pc-windows-gnu`, `i686-pc-windows-gnullvm` and
`x86_64-pc-windows-gnu` targets as long as LLVM-based C toolchains are used. Those include:

- [llvm-mingw][1]
- [MSYS2][2] with CLANG* environment

[1]: https://github.com/mstorsjo/llvm-mingw

[2]: https://www.msys2.org/docs/environments
