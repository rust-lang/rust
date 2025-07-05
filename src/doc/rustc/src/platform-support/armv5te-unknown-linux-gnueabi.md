# `armv5te-unknown-linux-gnueabi`

**Tier: 2**

This target supports Linux programs with glibc on ARMv5TE CPUs without
floating-point units.

## Target maintainers

There are currently no formally documented target maintainers.

## Requirements

The target is for cross-compilation only. Host tools are not supported.
std is fully supported.

## Building the target

Because this target is tier 2, artifacts are available from rustup.

## Building Rust programs

For building rust programs, you might want to specify GCC as linker in
`.cargo/config.toml` as follows:

```toml
[target.armv5te-unknown-linux-gnueabi]
linker = "arm-linux-gnueabi-gcc"
```
