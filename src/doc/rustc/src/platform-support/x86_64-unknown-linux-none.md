# `x86_64-unknown-linux-none`

**Tier: 3**

Freestanding x86-64 linux binary with no dependency on libc.

## Target maintainers

- [morr0ne](https://github.com/morr0ne/)

## Requirements

This target is cross compiled and can be built from any host.

This target has no support for host tools, std, or alloc.

## Building the target

The target can be built by enabling it for a `rustc` build:

```toml
[build]
build-stage = 1
target = ["x86_64-unknown-linux-none"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Created binaries will run on linux without any external requirements

## Cross-compilation toolchains and C code

Support for C code is currently untested
