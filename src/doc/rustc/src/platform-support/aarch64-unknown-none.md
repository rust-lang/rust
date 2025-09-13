# `aarch64-unknown-none` and `aarch64-unknown-none-softfloat`

* **Tier: 2**
* **Library Support:** core and alloc (bare-metal, `#![no_std]`)

Bare-metal targets for CPUs in the Armv8-A architecture family, running in AArch64 mode.

For the AArch32 mode carried over from Armv7-A, see
[`armv7a-none-eabi`](armv7a-none-eabi.md) instead.

Processors in this family include the [Arm Cortex-A35, 53, 76, etc][aarch64-cpus].

[aarch64-cpus]: https://en.wikipedia.org/wiki/Comparison_of_ARM_processors#ARMv8-A

## Target maintainers

[Rust Embedded Devices Working Group Arm Team]

[Rust Embedded Devices Working Group Arm Team]: https://github.com/rust-embedded/wg?tab=readme-ov-file#the-arm-team

## Target CPU and Target Feature options

All AArch64 processors include an FPU. The difference between the `-none` and
`-none-softfloat` targets is whether the FPU is used for passing function arguments.
You may prefer the `-softfloat` target when writing a kernel or interfacing with
pre-compiled binaries that use the soft-float ABI.

When using the hardfloat targets, the minimum floating-point features assumed
are those of the `fp-armv8`, which excludes NEON SIMD support. If your
processor supports a different set of floating-point features than the default
expectations of `fp-armv8`, then these should also be enabled or disabled as
needed with `-C target-feature=(+/-)`. It is also possible to tell Rust (or
LLVM) that you have a specific model of Arm processor, using the
[`-Ctarget-cpu`][target-cpu] option. Doing so may change the default set of
target-features enabled.

[target-cpu]: https://doc.rust-lang.org/rustc/codegen-options/index.html#target-cpu
[target-feature]: https://doc.rust-lang.org/rustc/codegen-options/index.html#target-feature

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

The *TrustedFirmware* group also maintain [Rust crates for this
target](https://github.com/ArmFirmwareCrates).

[`aarch64-cpu`]: https://docs.rs/aarch64-cpu
