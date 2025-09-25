# `armv8r-none-eabihf`

* **Tier: 3**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for CPUs in the Armv8-R architecture family, supporting
dual ARM/Thumb mode, with ARM mode as the default.

Processors in this family include the Arm [Cortex-R52][cortex-r52]
and [Cortex-R52+][cortex-r52-plus].

See [`arm-none-eabi`](arm-none-eabi.md) for information applicable to all
`arm-none-eabi` targets.

[cortex-r52]: https://www.arm.com/products/silicon-ip-cpu/cortex-r/cortex-r52
[cortex-r52-plus]: https://www.arm.com/products/silicon-ip-cpu/cortex-r/cortex-r52-plus

## Target maintainers

[@chrisnc](https://github.com/chrisnc)
[Rust Embedded Devices Working Group Arm Team]

[Rust Embedded Devices Working Group Arm Team]: https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team

## Requirements

The Cortex-R52 family always includes a floating-point unit, so there is no
non-`hf` version of this target. The floating-point features assumed by this
target are those of the single-precision-only config of the Cortex-R52, which
has 16 double-precision registers, accessible as 32 single-precision registers.
The other variant of Cortex-R52 includes double-precision, 32 double-precision
registers, and Advanced SIMD (Neon).

The manual refers to this as the "Full Advanced SIMD config". To compile code
for this variant, use: `-C target-feature=+fp64,+d32,+neon`. See the [Advanced
SIMD and floating-point support][fpu] section of the Cortex-R52 Processor
Technical Reference Manual for more details.

[fpu]: https://developer.arm.com/documentation/100026/0104/Advanced-SIMD-and-floating-point-support/About-the-Advanced-SIMD-and-floating-point-support

### Table of supported CPUs for `armv8r-none-eabihf`

| CPU         | FPU | Neon | Target CPU       | Target Features    |
|:----------- | --- |:---- |:---------------- |:------------------ |
| Any         | SP  | No   | None             | None               |
| Cortex-R52  | SP  | No   | `cortex-r52`     | `-fp64,-d32,-neon` |
| Cortex-R52  | DP  | No   | `cortex-r52`     | `-neon`            |
| Cortex-R52  | DP  | Yes  | `cortex-r52`     | None               |
| Cortex-R52+ | SP  | No   | `cortex-r52plus` | `-fp64,-d32,-neon` |
| Cortex-R52+ | DP  | No   | `cortex-r52plus` | `-neon`            |
| Cortex-R52+ | DP  | Yes  | `cortex-r52plus` | None               |
