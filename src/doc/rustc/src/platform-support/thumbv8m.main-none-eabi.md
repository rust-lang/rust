# `thumbv8m.main-none-eabi` and `thumbv8m.main-none-eabihf`

* **Tier: 2**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for CPUs in the Mainline [Armv8-M] architecture family,
supporting a subset of the [T32 ISA][t32-isa].

Processors in this family include the:

* [Arm Cortex-M33][cortex-m33]
* [Arm Cortex-M35P][cortex-m35p]

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets, in particular the difference between the `eabi` and
`eabihf` ABI.

[t32-isa]: https://developer.arm.com/Architectures/T32%20Instruction%20Set%20Architecture
[Armv8-M]: https://developer.arm.com/documentation/ddi0553/latest/
[cortex-m33]: https://developer.arm.com/Processors/Cortex-M33
[cortex-m35p]: https://developer.arm.com/Processors/Cortex-M35P

## Target maintainers

- [Rust Embedded Devices Working Group Arm Team](https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team)
- [arm-maintainers][arm_maintainers] ([rust@arm.com][arm_email])
    - Use `@rustbot ping arm-maintainers` to ping us

[arm_maintainers]: https://github.com/rust-lang/team/blob/master/teams/arm-maintainers.toml
[arm_email]: mailto:rust@arm.com

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

### Table of supported CPUs for `thumbv8m.main-none-eabihf`

| CPU         | FPU | DSP | MVE       | Target CPU    | Target Features       |
| ----------- | --- | --- | --------- | ------------- | --------------------- |
| Unspecified | SP  | No  | No        | None          | None                  |
| Cortex-M33  | SP  | No  | No        | `cortex-m33`  | `-dsp`                |
| Cortex-M33  | SP  | Yes | No        | `cortex-m33`  | None                  |
| Cortex-M33P | SP  | No  | No        | `cortex-m35p` | `-dsp`                |
| Cortex-M33P | SP  | Yes | No        | `cortex-m35p` | None                  |

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
