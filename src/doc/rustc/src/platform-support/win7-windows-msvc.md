# *-win7-windows-msvc

**Tier: 3**

Windows targets continuing support of windows7.

## Target maintainers

- @roblabla

## Requirements

This target supports full the entirety of std. This is automatically tested
every night on private infrastructure. Host tools may also work, though those
are not currently tested.

Those targets follow Windows calling convention for extern "C".

Like with any other Windows target created binaries are in PE format.

## Building the target

You can build Rust with support for the targets by adding it to the target list in config.toml:

```toml
[build]
build-stage = 1
target = [ "x86_64-win7-windows-msvc" ]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Created binaries work fine on Windows or Wine using native hardware.

## Cross-compilation toolchains and C code

Compatible C code can be built with either MSVC's `cl.exe` or LLVM's clang-cl.
