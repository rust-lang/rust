# `i686-unknown-hurd-gnu` and `x86_64-unknown-hurd-gnu`

**Tier: 3**

[GNU/Hurd] is the GNU Hurd is the GNU project's replacement for the Unix kernel.

## Target maintainers

[@sthibaul](https://github.com/sthibaul)

## Requirements

The target supports host tools.

The GNU/Hurd target supports `std` and uses the standard ELF file format.

## Building the target

This target can be built by adding `i686-unknown-hurd-gnu` and
`x86_64-unknown-hurd-gnu` as targets in the rustc list.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Tests can be run in the same way as a regular binary.

## Cross-compilation toolchains and C code

The target supports C code, the GNU toolchain calls the target
`i686-unknown-gnu` and `x86_64-unknown-gnu`.
