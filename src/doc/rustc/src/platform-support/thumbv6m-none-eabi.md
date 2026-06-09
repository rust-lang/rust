# `thumbv6m-none-eabi`

* **Tier: 2**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for CPUs in the [Armv6-M] architecture family, supporting a
subset of the [T32 ISA][t32-isa].

Processors in this family include the:

* [Arm Cortex-M0][cortex-m0]
* [Arm Cortex-M0+][cortex-m0plus]
* [Arm Cortex-M1][cortex-m1]

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

This target uses the soft-float ABI: functions which take `f32` or `f64` as
arguments will have those values packed into integer registers. This is the
only option because there is no FPU support in [Armv6-M].

[t32-isa]: https://developer.arm.com/Architectures/T32%20Instruction%20Set%20Architecture
[Armv6-M]: https://developer.arm.com/documentation/ddi0419/latest/
[cortex-m0]: https://developer.arm.com/Processors/Cortex-M0
[cortex-m0plus]: https://developer.arm.com/Processors/Cortex-M0+
[cortex-m1]: https://developer.arm.com/Processors/Cortex-M1

## Target maintainers

[Rust Embedded Devices Working Group Arm Team](https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team)

## Target CPU and Target Feature options

See [the bare-metal Arm
docs](arm-none-eabi.md#target-cpu-and-target-feature-options) for details on how
to use these flags.

### Table of supported CPUs

| CPU        | FPU | Target CPU      | Target Features       |
| ---------- | --- | --------------- | --------------------- |
| Cortex-M0  | No  | `cortex-m0`     | None                  |
| Cortex-M0+ | No  | `cortex-m0plus` | None                  |
| Cortex-M1  | No  | `cortex-m1`     | None                  |

### Arm Cortex-M0

The target CPU option is `cortex-m0`.

There are no relevant feature flags, and the FPU is not available.

### Arm Cortex-M0+

The target CPU option is `cortex-m0plus`.

There are no relevant feature flags, and the FPU is not available.

### Arm Cortex-M1

The target CPU option is `cortex-m1`.

There are no relevant feature flags, and the FPU is not available.
