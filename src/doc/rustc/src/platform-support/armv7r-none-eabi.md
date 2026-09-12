# `armv7r-none-eabi*` and `thumbv7r-none-eabi*`

* **Tier: 2**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for CPUs in the [Armv7-R] architecture family, supporting both
the [A32 (Arm) ISA][a32-isa] and [T32 (Thumb) ISA][t32-isa].

The `armv7r-none-eabi*` targets use A32 (Arm) mode by default and the
`thumbv7r-none-eabi*` targets use T32 (Thumb) mode by default.

Processors in this family include the:

* [Arm Cortex-R4][cortex-r4]
* [Arm Cortex-R5][cortex-r5]
* [Arm Cortex-R7][cortex-r7]
* [Arm Cortex-R8][cortex-r8]

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets, in particular the difference between the `eabi` and
`eabihf` ABI.

[t32-isa]: https://developer.arm.com/Architectures/T32%20Instruction%20Set%20Architecture
[a32-isa]: https://developer.arm.com/Architectures/A32%20Instruction%20Set%20Architecture
[Armv7-R]: https://support.arm.com/documentation/ddi0406
[cortex-r4]: https://developer.arm.com/Processors/Cortex-R4
[cortex-r5]: https://developer.arm.com/Processors/Cortex-R5
[cortex-r7]: https://developer.arm.com/Processors/Cortex-R7
[cortex-r8]: https://developer.arm.com/Processors/Cortex-R8

## Target maintainers

- [@chrisnc](https://github.com/chrisnc)
- [Rust Embedded Devices Working Group Arm Team]
- [arm-maintainers][arm_maintainers] ([rust@arm.com][arm_email])
    - Use `@rustbot ping arm-maintainers` to ping us

[Rust Embedded Devices Working Group Arm Team]: https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team
[arm_maintainers]: https://github.com/rust-lang/team/blob/master/teams/arm-maintainers.toml
[arm_email]: mailto:rust@arm.com

## Requirements

When using the hardfloat (`-eabibf`) targets, the minimum floating-point
features assumed are those of the `vfpv3-d16`, which includes single- and
double-precision, with 16 double-precision registers. See [VFP in the Cortex-R
processors][vfp] for more details on the possible FPU variants.

If your processor supports a different set of floating-point features than the
default expectations of `vfpv3-d16` (for example, if it only supports
single-precision and not double-precision), then those features should also be
enabled or disabled as needed with `-C target-feature=(+/-)` (or using a custom
JSON target). If you are removing features then you will also need to recompile
the Rust Standard Library from source (e.g. using `-Zbuild-std=core`).

See [the bare-metal Arm
docs](arm-none-eabi.md#target-cpu-and-target-feature-options) for details on how
to use these flags.

[vfp]: https://developer.arm.com/documentation/den0042/a/Floating-Point/Floating-point-basics-and-the-IEEE-754-standard/VFP-in-the-Cortex-R-processors


### Table of supported CPUs for `(arm|thumb)v7r-none-eabi`

| CPU       | FPU | Target CPU  | Target Features |
|-----------|-----|-------------|-----------------|
| Any       | No  | None        | None            |
| Cortex-R4 | No  | `cortex-r4` | None            |
| Cortex-R4 | DP  | `cortex-r4` | `+vfp3`         |
| Cortex-R4 | SP  | `cortex-r4` | `+vfp3,-fp64`   |
| Cortex-R5 | No  | `cortex-r5` | `-fpregs`       |
| Cortex-R5 | DP  | `cortex-r5` | None            |
| Cortex-R5 | SP  | `cortex-r5` | `-fp64`         |
| Cortex-R7 | No  | `cortex-r7` | `-fpregs`       |
| Cortex-R7 | DP  | `cortex-r7` | None            |
| Cortex-R7 | SP  | `cortex-r7` | `-fp64`         |
| Cortex-R8 | No  | `cortex-r8` | `-fpregs`       |
| Cortex-R8 | DP  | `cortex-r8` | None            |
| Cortex-R8 | SP  | `cortex-r8` | `-fp64`         |

### Table of supported CPUs for `(arm|thumb)v7r-none-eabihf`

| CPU       | FPU | Target CPU  | Target Features |
|-----------|-----|-------------|-----------------|
| Any       | DP  | None        | None            |
| Any       | SP  | None        | `-fp64`         |
| Cortex-R4 | DP  | `cortex-r4` | None            |
| Cortex-R4 | SP  | `cortex-r4` | `-fp64`         |
| Cortex-R5 | DP  | `cortex-r5` | None            |
| Cortex-R5 | SP  | `cortex-r5` | `-fp64`         |
| Cortex-R7 | DP  | `cortex-r7` | None            |
| Cortex-R7 | SP  | `cortex-r7` | `-fp64`         |
| Cortex-R8 | DP  | `cortex-r8` | None            |
| Cortex-R8 | SP  | `cortex-r8` | `-fp64`         |

<div class="warning">

Never use the `-fpregs` *target-feature* with the `(arm|thumb)v7r-none-eabi` targets
as it will cause compilation units to have different ABIs, which is unsound.

</div>

## Start-up and Low-Level Code

The [Rust Embedded Devices Working Group Arm Team] maintain the [`aarch32-cpu`]
and [`aarch32-rt`] crates, which may be useful for writing bare-metal code
using this target. Those crates include several examples which run in QEMU and
build using these targets.

[`aarch32-cpu`]: https://docs.rs/aarch32-cpu
[`aarch32-rt`]: https://docs.rs/aarch32-rt
