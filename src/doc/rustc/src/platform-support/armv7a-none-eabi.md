# `armv7a-none-eabi` and `armv7a-none-eabihf`

* **Tier: 2** for `armv7a-none-eabi`
* **Tier: 3** for `armv7a-none-eabihf`
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for CPUs in the Armv7-A architecture family, supporting
dual ARM/Thumb mode, with ARM mode as the default.

Note, this is for processors running in AArch32 mode. For the AArch64 mode
added in Armv8-A, see [`aarch64-unknown-none`](aarch64-unknown-none.md) instead.

Processors in this family include the [Arm Cortex-A5, 8, 32, etc][cortex-a].

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

[cortex-a]: https://en.wikipedia.org/wiki/ARM_Cortex-A

## Target maintainers

[Rust Embedded Devices Working Group Arm Team]

[Rust Embedded Devices Working Group Arm Team]: https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team

## Requirements

Almost all Armv7-A processors include an FPU (a VFPv3 or a VFPv4). The
difference between the `-eabi` and `-eabihf` targets is whether the FPU is
used for passing function arguments. You may prefer the `-eabi` soft-float
target when the processor does not have a floating point unit or the compiled
code should not use the floating point unit.

When using the hardfloat targets, the minimum floating-point features assumed
are those of the VFPv3-D16, which includes single- and double-precision, with
16 double-precision registers. This floating-point unit appears in Cortex-A8
and Cortex-A9 processors. See [VFP in the Cortex-A processors][vfp] for more
details on the possible FPU variants.

If your processor supports a different set of floating-point features than the
default expectations of VFPv3-D16, then these should also be enabled or
disabled as needed with `-C target-feature=(+/-)`.

In general, the following four combinations are possible:

- VFPv3-D16, target feature `+vfp3` and `-d32`
- VFPv3-D32, target feature `+vfp3` and `+d32`
- VFPv4-D16, target feature `+vfp4` and `-d32`
- VFPv4-D32, target feature `+vfp4` and `+d32`

An Armv7-A processor may optionally include a NEON hardware unit which
provides Single Instruction Multiple Data (SIMD) operations. The
implementation of this unit implies VFPv3-D32. The target feature `+neon` may
be added to inform the compiler about the availability of NEON.

You can refer to the [arm-none-eabi](arm-none-eabi.md) documentation for a
generic guide on target feature and target CPU specification and how to enable
and disable them via `.cargo/config.toml` file.

[vfp]: https://developer.arm.com/documentation/den0013/0400/Floating-Point/Floating-point-basics-and-the-IEEE-754-standard/ARM-VFP

## Start-up and Low-Level Code

The [Rust Embedded Devices Working Group Arm Team] maintain the [`cortex-ar`]
and [`cortex-a-rt`] crates, which may be useful for writing bare-metal code
using this target. The [`cortex-ar` repository](https://github.com/rust-embedded/cortex-ar)
includes several examples which run in QEMU and build using these targets.

[`cortex-ar`]: https://docs.rs/cortex-ar
[`cortex-a-rt`]: https://docs.rs/cortex-a-rt
