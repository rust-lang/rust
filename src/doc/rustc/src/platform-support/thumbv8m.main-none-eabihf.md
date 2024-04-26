# `thumbv8m.main-none-eabihf`

**Tier: 2**

Bare-metal target for CPUs in the Mainline [ARMv8-M] architecture family,
supporting a subset of the [T32 ISA][t32-isa].

Processors in this family include the:

* [Arm Cortex-M33F][cortex-m33]
* [Arm Cortex-M55F][cortex-m55]
* [Arm Cortex-M85F][cortex-m85]

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

This target uses the hard-float ABI: functions which take `f32` or `f64` as
arguments will have them passed via FPU registers. This target therefore
requires the use of an FPU (which is optional on Cortex-M33, Cortex-M55 and
Cortex-M85). See also the soft-float ABI version of this target
[`thumbv8m.main-none-eabi`](thumbv8m.main-none-eabi.md).

[t32-isa]: https://developer.arm.com/Architectures/T32%20Instruction%20Set%20Architecture
[ARMv8-M]: https://developer.arm.com/documentation/ddi0553/latest/
[cortex-m33]: https://developer.arm.com/Processors/Cortex-M33
[cortex-m55]: https://developer.arm.com/Processors/Cortex-M55
[cortex-m85]: https://developer.arm.com/Processors/Cortex-M85

## Target maintainers

* [Rust Embedded Devices Working Group Cortex-M
  Team](https://github.com/rust-embedded), `cortex-m@teams.rust-embedded.org`

## Target CPU and Target Feature options

See [the bare-metal Arm
docs](arm-none-eabi.md#target-cpu-and-target-feature-options) for details on how
to use these flags.

### Table of supported CPUs

| CPU         | FPU | DSP | MVE       | Target CPU    | Target Features       |
| ----------- | --- | --- | --------- | ------------- | --------------------- |
| Cortex-M33  | SP  | No  | N/A       | `cortex-m33`  | `-dsp`                |
| Cortex-M33  | SP  | Yes | N/A       | `cortex-m33`  | None                  |
| Cortex-M33P | SP  | No  | N/A       | `cortex-m35p` | `-dsp`                |
| Cortex-M33P | SP  | Yes | N/A       | `cortex-m35p` | None                  |
| Cortex-M55  | DP  | Yes | No        | `cortex-m55`  | `-mve`                |
| Cortex-M55  | DP  | Yes | Int       | `cortex-m55`  | `-mve.fp`             |
| Cortex-M55  | DP  | Yes | Int+Float | `cortex-m55`  | None                  |
| Cortex-M85  | DP  | Yes | No        | `cortex-m85`  | `-mve`                |
| Cortex-M85  | DP  | Yes | Int       | `cortex-m85`  | `-mve.fp`             |
| Cortex-M85  | DP  | Yes | Int+Float | `cortex-m85`  | None                  |

### Arm Cortex-M33

The target CPU is `cortex-m33`.

* Has optional DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target-cpu*
* Has an optional single precision FPU
  * support is enabled by default with this *target-cpu*
  * support is required when using the hard-float ABI

### Arm Cortex-M35P

The target CPU is `cortex-m35p`.

* Has optional DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target-cpu*
* Has a single precision FPU
  * support is enabled by default with this *target-cpu*
  * support is required when using the hard-float ABI

### Arm Cortex-M55

The target CPU is `cortex-m55`.

* Has DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target-cpu*
* Has an optional double-precision FPU that also supports half-precision FP16
  values
  * support is enabled by default with this *target-cpu*
  * support is required when using the hard-float ABI
* Has optional support for M-Profile Vector Extensions
  * Also known as *Helium Technology*
  * Available with only integer support, or both integer/float support
  * The appropriate feature for the MVE is either `mve` (integer) or `mve.fp`
    (float)
  * `mve.fp` is enabled by default on this target CPU
  * disable using `-mve.fp` (disable float MVE) or `-mve` (disable all MVE)

### Arm Cortex-M85

The target CPU is `cortex-m85`.

* Has DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target-cpu*
* Has an optional double-precision FPU that also supports half-precision FP16
  values
  * support is enabled by default with this *target-cpu*
  * support is required when using the hard-float ABI
* Has optional support for M-Profile Vector Extensions
  * Also known as *Helium Technology*
  * Available with only integer support, or both integer/float support
  * The appropriate feature for the MVE is either `mve` (integer) or `mve.fp`
    (float)
  * `mve.fp` is enabled by default on this target CPU
  * disable using `-mve.fp` (disable float MVE) or `-mve` (disable all MVE)
