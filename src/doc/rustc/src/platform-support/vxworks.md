# `*-wrs-vxworks`

**Tier: 3**

Targets for the VxWorks operating
system.

Target triplets available:

- `x86_64-wrs-vxworks`
- `aarch64-wrs-vxworks`
- `i686-wrs-vxworks`
- `armv7-wrs-vxworks-eabihf`
- `powerpc-wrs-vxworks`
- `powerpc64-wrs-vxworks`
- `powerpc-wrs-vxworks-spe`
- `riscv32-wrs-vxworks`
- `riscv64-wrs-vxworks`

## Target maintainers

- B I Mohammed Abbas ([@biabbas](https://github.com/biabbas))

## Requirements

### OS version

The minimum supported version is VxWorks 7.

## Building

Rust for each target can be cross-compiled with its specific target vsb configuration. Std support is added but not yet fully tested.

## Building the target

You can build Rust with support for the targets by adding it to the `target` list in `bootstrap.toml`. In addition the workbench and wr-cc have to configured and activated.

```toml
[build]
build-stage = 1
target = [
    "<HOST_TARGET>",
    "x86_64-wrs-vxworks",
    "aarch64-wrs-vxworks",
    "i686-wrs-vxworks",
    "armv7-wrs-vxworks-eabihf",
    "powerpc-wrs-vxworks",
    "powerpc64-wrs-vxworks",
    "powerpc-wrs-vxworks-spe",
]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for VxWorks.

The easiest way to build and test programs for VxWorks is to use the shipped rustc and cargo in VxWorks workbench, following the official windriver guidelines.

## Cross-compilation toolchains and C code

The target supports C code. Pre-compiled C toolchains can be found in provided VxWorks workbench.
