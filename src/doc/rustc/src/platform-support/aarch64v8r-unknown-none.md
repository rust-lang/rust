# `aarch64v8r-unknown-none` and `aarch64v8r-unknown-none-softfloat`

* **Tier: 3**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal target for CPUs in the Armv8-R architecture family, running in
AArch64 mode. Processors in this family include the
[Arm Cortex-R82][cortex-r82].

For Armv8-R CPUs running in AArch32 mode (such as the Arm Cortex-R52), see
[`armv8r-none-eabihf`](armv8r-none-eabihf.md) instead.

[cortex-r82]: https://developer.arm.com/processors/Cortex-R82

## Target maintainers

- [Rust Embedded Devices Working Group Arm Team]
- [@rust-lang/arm-maintainers][arm_maintainers] ([rust@arm.com][arm_email])

[Rust Embedded Devices Working Group Arm Team]: https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team
[arm_maintainers]: https://github.com/rust-lang/team/blob/master/teams/arm-maintainers.toml
[arm_email]: mailto:rust@arm.com

## Target CPU and Target Feature options

Unlike AArch64 v8-A processors, not all AArch64 v8-R processors include an FPU
(that is, not all Armv8-R AArch64 processors implement the optional Armv8
`FEAT_FP` extension). If you do not have an FPU, or have an FPU but wish to use
a soft-float ABI anyway, you should use the `aarch64v8r-unknown-none-softfloat`
target. If you wish to use the standard hard-float Arm AArch64 calling
convention, and you have an FPU, you can use the `aarch64v8r-unknown-none`
target.

When using the `aarch64v8r-unknown-none` target, the minimum floating-point
features assumed are the Advanced SIMD features (`FEAT_AdvSIMD`, or `+neon`),
the implementation of which is branded Arm NEON.

If your processor supports a different set of floating-point features than the
default expectations then these should also be enabled or disabled as needed
with [`-C target-feature=(+/-)`][target-feature]. However, note that currently
Rust does not support building hard-float AArch64 targets with Advanced SIMD
support disabled. It is also possible to tell Rust (or LLVM) that you have a
specific model of Arm processor, using the [`-Ctarget-cpu`][target-cpu] option.
Doing so may change the default set of target-features enabled.

[target-feature]: https://doc.rust-lang.org/rustc/codegen-options/index.html#target-feature
[target-cpu]: https://doc.rust-lang.org/rustc/codegen-options/index.html#target-cpu

## Requirements

These targets are cross-compiled and use static linking.

By default, the `lld` linker included with Rust will be used; however, you may
want to use the GNU linker instead. This can be obtained for Windows/Mac/Linux
from the [Arm Developer Website][arm-gnu-toolchain], or possibly from your OS's
package manager. To use it, add the following to your `.cargo/config.toml`:

```toml
[target.aarch64-unknown-none]
linker = "aarch64-none-elf-ld"
```

The GNU linker can also be used by specifying `aarch64-none-elf-gcc` as the
linker. This is needed when using GCC's link time optimization.

These targets don't provide a linker script, so you'll need to bring your own
according to the specific device you are using. Pass
`-Clink-arg=-Tyour_script.ld` as a rustc argument to make the linker use
`your_script.ld` during linking.

[arm-gnu-toolchain]: https://developer.arm.com/Tools%20and%20Software/GNU%20Toolchain

## Cross-compilation toolchains and C code

This target supports C code compiled with the `aarch64-none-elf` target
triple and a suitable `-march` or `-mcpu` flag.

## Start-up and Low-Level Code

The [Rust Embedded Devices Working Group Arm Team] maintain the
[`aarch64-cpu`] crate, which may be useful for writing bare-metal code using
this target.

[`aarch64-cpu`]: https://docs.rs/aarch64-cpu
