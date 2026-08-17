# `*-oe-linux-gnu`

**Tier: 3**

Targets for OpenEmbedded/Yocto-based Linux systems.

Target triplets available:

* `x86_64-oe-linux-gnu`
* `aarch64-oe-linux-gnu`
* `i686-oe-linux-gnu`
* `armv7-oe-linux-gnueabihf`
* `riscv64-oe-linux-gnu`

## Target maintainers

[@DeepeshWR](https://github.com/DeepeshWR)

## Requirements

### Operating System

These targets are intended for Linux systems built using the OpenEmbedded and Yocto Project build systems.

### C Toolchain

The targets use the architecture-specific OpenEmbedded GCC driver as the default linker and therefore require the corresponding OpenEmbedded cross-compilation toolchain to be available in the build environment.

## Building

These targets are intended for use with the OpenEmbedded/Yocto build system. They provide base target definitions from which downstream Yocto-generated targets may inherit.

## Building Rust programs

Rust does not currently ship precompiled artifacts for these targets.

Downstream Yocto-generated targets may inherit from these base OpenEmbedded targets and customize metadata or vendor-specific configuration as needed.

## Cross-compilation toolchains and C code

The targets support linking with C code through the corresponding OpenEmbedded cross-compilation toolchains.
