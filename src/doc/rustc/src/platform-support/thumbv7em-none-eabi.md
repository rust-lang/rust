# `thumbv7em-none-eabi` and `thumbv7em-none-eabihf`

* **Tier: 2**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for CPUs in the [Armv7E-M] architecture family, supporting a
subset of the [T32 ISA][t32-isa].

Processors in this family include the:

* [Arm Cortex-M4][cortex-m4] and Arm Cortex-M4F
* [Arm Cortex-M7][cortex-m7] and Arm Cortex-M7F

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets, in particular the difference between the `eabi` and
`eabihf` ABI.

[t32-isa]: https://developer.arm.com/Architectures/T32%20Instruction%20Set%20Architecture
[Armv7E-M]: https://developer.arm.com/documentation/ddi0403/latest/
[cortex-m4]: https://developer.arm.com/Processors/Cortex-M4
[cortex-m7]: https://developer.arm.com/Processors/Cortex-M7

## Target maintainers

[Rust Embedded Devices Working Group Arm Team](https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team)

## Target CPU and Target Feature options

See [the bare-metal Arm
docs](arm-none-eabi.md#target-cpu-and-target-feature-options) for details on how
to use these flags.

### Table of supported CPUs for `thumbv7em-none-eabi`

| CPU        | FPU | DSP | Target CPU  | Target Features |
| ---------- | --- | --- | ----------- | --------------- |
| Any        | No  | Yes | None        | None            |
| Cortex-M4  | No  | Yes | `cortex-m4` | `-fpregs`       |
| Cortex-M4F | SP  | Yes | `cortex-m4` | None            |
| Cortex-M7  | No  | Yes | `cortex-m7` | `-fpregs`       |
| Cortex-M7F | SP  | Yes | `cortex-m7` | `-fp64`         |
| Cortex-M7F | DP  | Yes | `cortex-m7` | None            |

### Table of supported CPUs for `thumbv7em-none-eabihf`

| CPU        | FPU | DSP | Target CPU  | Target Features |
| ---------- | --- | --- | ----------- | --------------- |
| Any        | SP  | Yes | None        | None            |
| Cortex-M4F | SP  | Yes | `cortex-m4` | None            |
| Cortex-M7F | SP  | Yes | `cortex-m7` | `-fp64`         |
| Cortex-M7F | DP  | Yes | `cortex-m7` | None            |

<div class="warning">

Never use the `-fpregs` *target-feature* with the `thumbv7em-none-eabihf` target
as it will cause compilation units to have different ABIs, which is unsound.

</div>

### Arm Cortex-M4 and Arm Cortex-M4F

The target CPU is `cortex-m4`.

* All Cortex-M4 have DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target*
* Cortex-M4F has a single precision FPU
  * support is enabled by default with this *target-cpu*
  * disable support using the `-fpregs` *target-feature* (`eabi` only)

### Arm Cortex-M7 and Arm Cortex-M7F

The target CPU is `cortex-m7`.

* All Cortex-M7 have DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target*
* Cortex-M7F have either a single-precision or double-precision FPU
  * double-precision support is enabled by default with this *target-cpu*
    * opt-out by using the `-f64` *target-feature*
  * disable support entirely using the `-fpregs` *target-feature* (`eabi` only)
