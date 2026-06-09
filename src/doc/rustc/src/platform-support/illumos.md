# `aarch64-unknown-illumos` and `x86_64-unknown-illumos`

**Tier: 2/3**

[illumos](https://www.illumos.org/), is a Unix operating system which provides next-generation features for downstream distributions,
including advanced system debugging, next generation filesystem, networking, and virtualization options.

## Target maintainers

[@jclulow](https://github.com/jclulow)
[@pfmooney](https://github.com/pfmooney)

## Requirements

The target supports host tools.

The illumos target supports `std` and uses the standard ELF file format.

`x86_64-unknown-illumos` is a tier 2 target with host tools.
`aarch64-unknown-illumos` is a tier 3 target.

## Building the target

These targets can be built by adding `aarch64-unknown-illumos` and
`x86_64-unknown-illumos` as targets in the rustc list.

## Building Rust programs

Rust ships pre-compiled artifacts for the `x86_64-unknown-illumos` target.
Rust does not ship pre-compiled artifacts for `aarch64-unknown-illumos`,
it requires building the target either as shown above or using `-Zbuild-std`.

## Testing

Tests can be run in the same way as a regular binary.

## Cross-compilation toolchains and C code

The target supports C code.

The illumos project makes available [prebuilt sysroot artefacts](https://github.com/illumos/sysroot) which can be used for cross compilation.
The official Rust binaries are cross-compiled using these artefacts.
