# powerpc-unknown-linux-muslspe

**Tier: 3**

This target is very similar to already existing ones like `powerpc-unknown-linux-musl` and `powerpc-unknown-linux-gnuspe`.
This one has PowerPC SPE support for musl. Unfortunately, the last supported gcc version with PowerPC SPE is 8.4.0.

See also [platform support documentation of `powerpc-unknown-linux-gnuspe`](powerpc-unknown-linux-gnuspe.md) for information about PowerPC SPE.

## Target maintainers

[@BKPepe](https://github.com/BKPepe)

## Requirements

This target is cross-compiled. There is no support for `std`. There is no
default allocator, but it's possible to use `alloc` by supplying an allocator.

This target generated binaries in the ELF format.

## Building the target

This target was tested and used within the `OpenWrt` build system for CZ.NIC Turris 1.x routers using Freescale P2020.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

This is a cross-compiled target and there is no support to run rustc test suite.
