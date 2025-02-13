# `x86_64-pc-cygwin`

**Tier: 3**

Windows targets supporting Cygwin.
The `*-cygwin` targets are **not** intended as native target for applications,
a developer writing Windows applications should use the `*-pc-windows-*` targets instead, which are *native* Windows.

Cygwin is only intended as an emulation layer for Unix-only programs which do not support the native Windows targets.

## Target maintainers

- [Berrysoft](https://github.com/Berrysoft)

## Requirements

This target is cross compiled. It needs `x86_64-pc-cygwin-gcc` as linker.

The `target_os` of the target is `cygwin`, and it is `unix`.

## Building the target

For cross-compilation you want LLVM with [llvm/llvm-project#121439 (merged)](https://github.com/llvm/llvm-project/pull/121439) applied to fix the LLVM codegen on importing external global variables from DLLs.
No native builds on Cygwin now. It should be possible theoretically though, but might need a lot of patches.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Created binaries work fine on Windows with Cygwin.

## Cross-compilation toolchains and C code

Compatible C code can be built with GCC shipped with Cygwin. Clang is untested.
