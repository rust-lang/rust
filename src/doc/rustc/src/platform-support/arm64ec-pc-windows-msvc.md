# `arm64ec-pc-windows-msvc`

**Tier: 2**

Arm64EC ("Emulation Compatible") for mixed architecture (AArch64 and x86_64)
applications on AArch64 Windows 11. See <https://learn.microsoft.com/en-us/windows/arm/arm64ec>.

## Target maintainers

- [@dpaoliello](https://github.com/dpaoliello)

## Requirements

Builds Arm64EC static and dynamic libraries and executables which can be run on
AArch64 Windows 11 devices. Arm64EC static libraries can also be linked into
Arm64X dynamic libraries and executables.

Only supported backend is LLVM 18 or above:
* 18.1.0 added initial support for Arm64EC.
* 18.1.2 fixed import library generation (required for `raw-dylib` support).
* 18.1.4 fixed linking issue for some intrinsics implemented in
  `compiler_builtins`.

Visual Studio 2022 (or above) with the "ARM64/ARM64EC built tools" component and
the Windows 11 SDK are required.

### Reusing code from other architectures - x86_64 or AArch64?

Arm64EC uses `arm64ec` as its `target_arch`, but it is possible to reuse
existing architecture-specific code in most cases. The best mental model for
deciding which architecture to reuse is to is to think of Arm64EC as an x86_64
process that happens to use the AArch64 instruction set (with some caveats) and
has a completely custom ABI.

To put this in practice:
* Arm64EC interacts with the operating system, other processes and other DLLs as
  x86_64.
  - For example, [in `backtrace`](https://github.com/rust-lang/backtrace-rs/commit/ef39a7d7da58b4cae8c8f3fc67a8300fd8d2d0d9)
    we use the x86_64 versions of `CONTEXT` and `RtlVirtualUnwind`.
  - If you are configuring a search path to find DLLs (e.g., to load plugins or
    addons into your application), you should use the same path as the x86_64
    version of your application, not the AArch64 path (since Arm64EC (i.e.,
    x86_64) processes cannot load native AArch64 DLLs).
* Arm64EC uses AArch64 intrinsics.
  - For example, <https://github.com/rust-lang/portable-simd/commit/ca4033f49b1f6019561b8b161b4097b4a07f2e1b>
    and <https://github.com/rust-lang/stdarch/commit/166ef7ba22b6a1d908d4b29a36e68ceca324808a>.
* Assembly for AArch64 might be reusable for Arm64EC, but there are many
  caveats. For full details see [Microsoft's documentation on the Arm64EC ABI](https://learn.microsoft.com/en-us/windows/arm/arm64ec-abi)
  but in brief:
  - Arm64EC uses a subset of AArch64 registers.
  - Arm64EC uses a different name mangling scheme than AArch64.
  - Arm64EC requires entry and exit thunks be generated for some functions.
  - Indirect calls must be done via a call checker.
  - Control Flow Guard and stack checks use different functions than AArch64.

## Building the target

You can build Rust with support for the targets by adding it to the `target`
list in `bootstrap.toml`:

```toml
[build]
target = ["arm64ec-pc-windows-msvc"]
```

## Building Rust programs

These targets are distributed through `rustup`, and otherwise require no
special configuration.

## Testing

Tests can be run on AArch64 Windows 11 devices.

## Cross-compilation toolchains and C code

C code can be built using the Arm64-targeting MSVC or Clang toolchain.

To compile:

```bash
cl /arm64EC /c ...
```

To link:

```bash
link /MACHINE:ARM64EC ...
```

Further reading: <https://learn.microsoft.com/en-us/windows/arm/arm64ec-build>
