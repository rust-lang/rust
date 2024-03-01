# `arm64ec-pc-windows-msvc`

**Tier: 3**

Arm64EC ("Emulation Compatible") for mixed architecture (AArch64 and x86_64)
applications on AArch64 Windows 11. See <https://learn.microsoft.com/en-us/windows/arm/arm64ec>.

## Target maintainers

- [@dpaoliello](https://github.com/dpaoliello)

## Requirements

Target only supports cross-compilation, `core` and `alloc` are supported but
`std` is not.

Builds Arm64EC static and dynamic libraries and executables which can be run on
AArch64 Windows 11 devices. Arm64EC static libraries can also be linked into
Arm64X dynamic libraries and executables.

Uses `arm64ec` as its `target_arch` - code built for Arm64EC must be compatible
with x86_64 code (e.g., same structure layouts, function signatures, etc.) but
use AArch64 intrinsics.

Only supported backend is LLVM 18 (or above).

## Building the target

You can build Rust with support for the targets by adding it to the `target`
list in `config.toml` and disabling `std`:

```toml
[build]
target = [ "arm64ec-pc-windows-msvc" ]

[target.arm64ec-pc-windows-msvc]
no-std = true
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Tests can be run on AArch64 Windows 11 devices.

Since this is a `no_std` target, the Rust test suite is not supported.

## Cross-compilation toolchains and C code

C code can be built using the Arm64-targetting MSVC toolchain.

To compile:

```bash
cl /arm64EC /c ...
```

To link:

```bash
link /MACHINE:ARM64EC ...
```

Further reading: <https://learn.microsoft.com/en-us/windows/arm/arm64ec-build>
