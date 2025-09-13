# `armv7r-none-eabi` and `armv7r-none-eabihf`

* **Tier: 2**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for CPUs in the Armv7-R architecture family, supporting
dual ARM/Thumb mode, with ARM mode as the default.

Processors in this family include the [Arm Cortex-R4, 5, 7, and 8][cortex-r].

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

[cortex-r]: https://en.wikipedia.org/wiki/ARM_Cortex-R

## Target maintainers

[@chrisnc](https://github.com/chrisnc)
[Rust Embedded Devices Working Group Arm Team]

[Rust Embedded Devices Working Group Arm Team]: https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team

## Requirements

When using the hardfloat targets, the minimum floating-point features assumed
are those of the `vfpv3-d16`, which includes single- and double-precision, with
16 double-precision registers. This floating-point unit appears in Cortex-R4F
and Cortex-R5F processors. See [VFP in the Cortex-R processors][vfp]
for more details on the possible FPU variants.

If your processor supports a different set of floating-point features than the
default expectations of `vfpv3-d16`, then these should also be enabled or
disabled as needed with `-C target-feature=(+/-)`.

[endianness]: https://developer.arm.com/documentation/den0042/a/Coding-for-Cortex-R-Processors/Endianness

[vfp]: https://developer.arm.com/documentation/den0042/a/Floating-Point/Floating-point-basics-and-the-IEEE-754-standard/VFP-in-the-Cortex-R-processors

## Start-up and Low-Level Code

The [Rust Embedded Devices Working Group Arm Team] maintain the [`cortex-ar`]
and [`cortex-r-rt`] crates, which may be useful for writing bare-metal code
using this target. Those crates include several examples which run in QEMU and
build using these targets.

[`cortex-ar`]: https://docs.rs/cortex-ar
[`cortex-r-rt`]: https://docs.rs/cortex-r-rt
