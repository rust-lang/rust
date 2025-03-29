# `mips64-openwrt-linux-musl`
**Tier: 3**

## Target maintainers

[@Itus-Shield](https://github.com/Itus-Shield)

## Requirements
This target is cross-compiled. There is no support for `std`. There is no
default allocator, but it's possible to use `alloc` by supplying an allocator.

By default, Rust code generated for this target uses `-msoft-float` and is
dynamically linked.

This target generated binaries in the ELF format.

## Building the target
This target is built exclusively within the `OpenWrt` build system via
the `rust-lang` HOST package

## Building Rust programs
Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above).

## Testing
As `mips64-openwrt-linux-musl` supports a variety of different environments and does
not support `std`, this target does not support running the Rust testsuite at this
time.
