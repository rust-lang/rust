# `thumbv7em-none-eabihf`

**Tier: 2**

Bare-metal target for CPUs in the [ARMv7E-M] architecture family that have an
FPU, supporting a subset of the [T32 ISA][t32-isa].

Processors in this family include the:

* [Arm Cortex-M4F][cortex-m4]
* [Arm Cortex-M7F][cortex-m7]

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

This target uses the hard-float ABI: functions which take `f32` or `f64` as
arguments will have them passed via FPU registers. This target therefore
requires the use of an FPU (which is optional on Cortex-M4 and Cortex-M7). See
also the soft-float ABI version of this target
[`thumbv7em-none-eabi`](thumbv7em-none-eabi.md).

[t32-isa]: https://developer.arm.com/Architectures/T32%20Instruction%20Set%20Architecture
[ARMv7E-M]: https://developer.arm.com/documentation/ddi0403/latest/
[cortex-m4]: https://developer.arm.com/Processors/Cortex-M4
[cortex-m7]: https://developer.arm.com/Processors/Cortex-M7

## Target maintainers

* [Rust Embedded Devices Working Group Cortex-M
  Team](https://github.com/rust-embedded), `cortex-m@teams.rust-embedded.org`

## Target CPU and Target Feature options

See [the bare-metal Arm
docs](arm-none-eabi.md#target-cpu-and-target-feature-options) for details on how
to use these flags.

### Table of supported CPUs

| CPU        | FPU | DSP | Target CPU  | Target Features |
| ---------- | --- | --- | ----------- | --------------- |
| Cortex-M4F | SP  | Yes | `cortex-m4` | None            |
| Cortex-M7F | SP  | Yes | `cortex-m7` | `-fp64`         |
| Cortex-M7F | DP  | Yes | `cortex-m7` | None            |

### Arm Cortex-M4 and Arm Cortex-M4F

The target CPU is `cortex-m4`.

* All Cortex-M4 have DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target-cpu*
* Cortex-M4F has a single precision FPU
  * support is enabled by default with this *target*
  * support is required when using the hard-float ABI

### Arm Cortex-M7 and Arm Cortex-M7F

The target CPU is `cortex-m7`.

* All Cortex-M7 have DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target-cpu*
* Cortex-M7F have either a single-precision or double-precision FPU
  * single precision support is enabled by default with this *target*
  * double-precision support is enabled by default with this *target-cpu*
    * opt-out by using the `-f64` *target-feature*
