# `thumbv8m.main-none-eabi` and `thumbv8m.main-none-eabihf`

* **Tier: 2**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for CPUs in the Mainline [Armv8-M] architecture family,
supporting a subset of the [T32 ISA][t32-isa].

Processors in this family include the:

* [Arm Cortex-M33][cortex-m33]
* [Arm Cortex-M35P][cortex-m35p]
* [Arm Cortex-M55][cortex-m55]
* [Arm Cortex-M85][cortex-m85]

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets, in particular the difference between the `eabi` and
`eabihf` ABI.

[t32-isa]: https://developer.arm.com/Architectures/T32%20Instruction%20Set%20Architecture
[Armv8-M]: https://developer.arm.com/documentation/ddi0553/latest/
[cortex-m33]: https://developer.arm.com/Processors/Cortex-M33
[cortex-m35p]: https://developer.arm.com/Processors/Cortex-M35P
[cortex-m55]: https://developer.arm.com/Processors/Cortex-M55
[cortex-m85]: https://developer.arm.com/Processors/Cortex-M85

## Target maintainers

[Rust Embedded Devices Working Group Arm Team](https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team)

## Target CPU and Target Feature options

See [the bare-metal Arm
docs](arm-none-eabi.md#target-cpu-and-target-feature-options) for details on how
to use these flags.

### Table of supported CPUs for `thumbv8m.main-none-eabi`

| CPU         | FPU | DSP | MVE       | Target CPU    | Target Features       |
| ----------- | --- | --- | --------- | ------------- | --------------------- |
| Unspecified | No  | No  | No        | None          | None                  |
| Cortex-M33  | No  | No  | No        | `cortex-m33`  | `-fpregs,-dsp`        |
| Cortex-M33  | No  | Yes | No        | `cortex-m33`  | `-fpregs`             |
| Cortex-M33  | SP  | No  | No        | `cortex-m33`  | `-dsp`                |
| Cortex-M33  | SP  | Yes | No        | `cortex-m33`  | None                  |
| Cortex-M35P | No  | No  | No        | `cortex-m35p` | `-fpregs,-dsp`        |
| Cortex-M35P | No  | Yes | No        | `cortex-m35p` | `-fpregs`             |
| Cortex-M35P | SP  | No  | No        | `cortex-m35p` | `-dsp`                |
| Cortex-M35P | SP  | Yes | No        | `cortex-m35p` | None                  |
| Cortex-M55  | No  | Yes | No        | `cortex-m55`  | `-fpregs,-mve`        |
| Cortex-M55  | DP  | Yes | No        | `cortex-m55`  | `-mve`                |
| Cortex-M55  | No  | Yes | Int       | `cortex-m55`  | `-fpregs,-mve.fp,+mve`|
| Cortex-M55  | DP  | Yes | Int       | `cortex-m55`  | `-mve.fp`             |
| Cortex-M55  | DP  | Yes | Int+Float | `cortex-m55`  | None                  |
| Cortex-M85  | No  | Yes | No        | `cortex-m85`  | `-fpregs,-mve`        |
| Cortex-M85  | DP  | Yes | No        | `cortex-m85`  | `-mve`                |
| Cortex-M85  | No  | Yes | Int       | `cortex-m85`  | `-fpregs,-mve.fp,+mve`|
| Cortex-M85  | DP  | Yes | Int       | `cortex-m85`  | `-mve.fp`             |
| Cortex-M85  | DP  | Yes | Int+Float | `cortex-m85`  | None                  |

### Table of supported CPUs for `thumbv8m.main-none-eabihf`

| CPU         | FPU | DSP | MVE       | Target CPU    | Target Features       |
| ----------- | --- | --- | --------- | ------------- | --------------------- |
| Unspecified | SP  | No  | No        | None          | None                  |
| Cortex-M33  | SP  | No  | No        | `cortex-m33`  | `-dsp`                |
| Cortex-M33  | SP  | Yes | No        | `cortex-m33`  | None                  |
| Cortex-M33P | SP  | No  | No        | `cortex-m35p` | `-dsp`                |
| Cortex-M33P | SP  | Yes | No        | `cortex-m35p` | None                  |
| Cortex-M55  | DP  | Yes | No        | `cortex-m55`  | `-mve`                |
| Cortex-M55  | DP  | Yes | Int       | `cortex-m55`  | `-mve.fp`             |
| Cortex-M55  | DP  | Yes | Int+Float | `cortex-m55`  | None                  |
| Cortex-M85  | DP  | Yes | No        | `cortex-m85`  | `-mve`                |
| Cortex-M85  | DP  | Yes | Int       | `cortex-m85`  | `-mve.fp`             |
| Cortex-M85  | DP  | Yes | Int+Float | `cortex-m85`  | None                  |

*Technically* you can use this hard-float ABI on a CPU which has no FPU but does
have Integer MVE, because MVE provides the same set of registers as the FPU
(including `s0` and `d0`). The particular set of flags that might enable this
unusual scenario are currently not recorded here.

<div class="warning">

Never use the `-fpregs` *target-feature* with the `thumbv8m.main-none-eabihf`
target as it will cause compilation units to have different ABIs, which is
unsound.

</div>

### Arm Cortex-M33

The target CPU is `cortex-m33`.

* Has optional DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target-cpu*
* Has an optional single precision FPU
  * support is enabled by default with this *target-cpu*
  * disable support using the `-fpregs` *target-feature* (`eabi` only)

### Arm Cortex-M35P

The target CPU is `cortex-m35p`.

* Has optional DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target-cpu*
* Has an optional single precision FPU
  * support is enabled by default with this *target-cpu*
  * disable support using the `-fpregs` *target-feature* (`eabi` only)

### Arm Cortex-M55

The target CPU is `cortex-m55`.

* Has DSP extensions
  * support is controlled by the `dsp` *target-feature*
  * enabled by default with this *target-cpu*
* Has an optional double-precision FPU that also supports half-precision FP16
  values
  * support is enabled by default with this *target-cpu*
  * disable support using the `-fpregs` *target-feature* (`eabi` only)
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
  * disable support using the `-fpregs` *target-feature* (`eabi` only)
* Has optional support for M-Profile Vector Extensions
  * Also known as *Helium Technology*
  * Available with only integer support, or both integer/float support
  * The appropriate feature for the MVE is either `mve` (integer) or `mve.fp`
    (float)
  * `mve.fp` is enabled by default on this target CPU
  * disable using `-mve.fp` (disable float MVE) or `-mve` (disable all MVE)
