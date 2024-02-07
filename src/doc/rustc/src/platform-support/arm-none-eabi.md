# `{arm,thumb}*-none-eabi(hf)?`

**Tier: 2**
- [arm(eb)?v7r-none-eabi(hf)?](armv7r-none-eabi.md)
- armv7a-none-eabi
- thumbv6m-none-eabi
- thumbv7m-none-eabi
- thumbv7em-none-eabi(hf)?
- thumbv8m.base-none-eabi
- thumbv8m.main-none-eabi(hf)?

**Tier: 3**
- [{arm,thumb}v4t-none-eabi](armv4t-none-eabi.md)
- [{arm,thumb}v5te-none-eabi](armv5te-none-eabi.md)
- armv7a-none-eabihf
- [armv8r-none-eabihf](armv8r-none-eabihf.md)

Bare-metal target for 32-bit ARM CPUs.

If a target has a `*hf` variant, that variant uses the hardware floating-point
ABI and enables some minimum set of floating-point features based on the FPU(s)
available in that processor family.

## Requirements

These targets are cross-compiled and use static linking.

By default, the `lld` linker included with Rust will be used; however, you may
want to use the GNU linker instead. This can be obtained for Windows/Mac/Linux
from the [Arm Developer Website][arm-gnu-toolchain], or possibly from your OS's
package manager. To use it, add the following to your `.cargo/config.toml`:

```toml
[target.<your-target>]
linker = "arm-none-eabi-ld"
```

The GNU linker can also be used by specifying `arm-none-eabi-gcc` as the
linker. This is needed when using GCC's link time optimization.

[arm-gnu-toolchain]: https://developer.arm.com/Tools%20and%20Software/GNU%20Toolchain

These targets don't provide a linker script, so you'll need to bring your own
according to the specific device you are using. Pass
`-Clink-arg=-Tyour_script.ld` as a rustc argument to make the linker use
`your_script.ld` during linking.

Targets named `thumb*` instead of `arm*`
generate Thumb-mode code by default. M-profile processors (`thumbv*m*-*`
targets) only support Thumb-mode code.
For the `arm*` targets, Thumb-mode code generation can be enabled by using
`-C target-feature=+thumb-mode`. Using the unstable
`#![feature(arm_target_feature)]`, the attribute
`#[target_feature(enable = "thumb-mode")]` can be applied to individual
`unsafe` functions to cause those functions to be compiled to Thumb-mode code.

## Building Rust Programs

For the Tier 3 targets in this family, rust does not ship pre-compiled
artifacts.

Just use the `build-std` nightly cargo feature to build the `core` library. You
can pass this as a command line argument to cargo, or your `.cargo/config.toml`
file might include the following lines:

```toml
[unstable]
build-std = ["core"]
```

Most of `core` should work as expected, with the following notes:
* If the target is not `*hf`, then floating-point operations are emulated in
  software.
* Integer division is also emulated in software on some targets, depending on
  the CPU.
* Architectures prior to ARMv7 don't have atomic instructions.

`alloc` is also supported, as long as you provide your own global allocator.

Rust programs are output as ELF files.

## Testing

This is a cross-compiled target that you will need to emulate during testing.

The exact emulator that you'll need depends on the specific device you want to
run your code on.

## Cross-compilation toolchains and C code

The target supports C code compiled with the `arm-none-eabi` target triple and
a suitable `-march` or `-mcpu` flag.

`gcc` or `clang` can be used, but note that `gcc` uses `-fshort-enums` by
default for `arm-none*` targets, while `clang` does not. `rustc` matches the
`gcc` behavior, i.e., the size of a `#[repr(C)] enum` in Rust can be as little
as 1 byte, rather than 4, as they are on `arm-linux` targets.
