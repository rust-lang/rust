# `arm(eb)?v7r-none-eabi(hf)?`

**Tier: 2**

Bare-metal target for CPUs in the Armv7-R architecture family, supporting
dual ARM/Thumb mode, with ARM mode as the default.

Processors in this family include the [Arm Cortex-R4, 5, 7, and 8][cortex-r].

The `eb` versions of this target generate code for big-endian processors.

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

[cortex-r]: https://en.wikipedia.org/wiki/ARM_Cortex-R

## Target maintainers

[@chrisnc](https://github.com/chrisnc)

## Requirements

When using the big-endian version of this target, note that some variants of
the Cortex-R have both big-endian instructions and data. This configuration is
known as BE-32, while data-only big-endianness is known as BE-8. To build
programs for BE-32 processors, the GNU linker must be used with the `-mbe32`
option. See [ARM Cortex-R Series Programmer's Guide: Endianness][endianness]
for more details about different endian modes.

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

## Cross-compilation toolchains and C code

This target supports C code compiled with the `arm-none-eabi` target triple and
`-march=armv7-r` or a suitable `-mcpu` flag.
