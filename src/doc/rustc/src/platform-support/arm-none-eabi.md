# `{arm,thumb}*-none-eabi(hf)?`

## Tier 2 Target List

- Arm A-Profile Architectures
  - `armv7a-none-eabi`
- Arm R-Profile Architectures
  - [`armv7r-none-eabi` and `armv7r-none-eabihf`](armv7r-none-eabi.md)
  - [`armebv7r-none-eabi` and `armebv7r-none-eabihf`](armv7r-none-eabi.md)
- Arm M-Profile Architectures
  - [`thumbv6m-none-eabi`](thumbv6m-none-eabi.md)
  - [`thumbv7m-none-eabi`](thumbv7m-none-eabi.md)
  - [`thumbv7em-none-eabi`](thumbv7em-none-eabi.md) and [`thumbv7em-none-eabihf`](thumbv7em-none-eabihf.md)
  - [`thumbv8m.base-none-eabi`](thumbv8m.base-none-eabi.md)
  - [`thumbv8m.main-none-eabi`](thumbv8m.main-none-eabi.md) and [`thumbv8m.main-none-eabihf`](thumbv8m.main-none-eabihf.md)
- *Legacy* Arm Architectures
  - None

## Tier 3 Target List

- Arm A-Profile Architectures
  - `armv7a-none-eabihf`
- Arm R-Profile Architectures
  - [`armv8r-none-eabihf`](armv8r-none-eabihf.md)
- Arm M-Profile Architectures
  - None
- *Legacy* Arm Architectures
  - [`armv4t-none-eabi` and `thumbv4t-none-eabi`](armv4t-none-eabi.md)
  - [`armv5te-none-eabi` and `thumbv5te-none-eabi`](armv5te-none-eabi.md)

## Common Target Details

This documentation covers details that apply to a range of bare-metal target for
32-bit ARM CPUs. In addition, target specific details may be covered in their
own document.

If a target ends in `eabi`, that target uses the so-called *soft-float ABI*:
functions which take `f32` or `f64` as arguments will have those values packed
into integer registers. This means that an FPU is not required from an ABI
perspective, but within a function FPU instructions may still be used if the
code is compiled with a `target-cpu` or `target-feature` option that enables
FPU support.

If a target ends if `eabihf`, that target uses the so-called *hard-float ABI*:
functions which take `f32` or `f64` as arguments will have them passed via FPU
registers. These targets therefore require the use of an FPU and will assume the
minimum support FPU for that architecture is available. More advanced FPU
instructions (e.g. ones that work on double-precision `f64` values) may be
generated if the code is compiled with a `target-cpu` or `target-feature` option
that enables such additional FPU support.

## Target CPU and Target Feature options

It is possible to tell Rust (or LLVM) that you have a specific model of Arm
processor, using the [`-C target-cpu`][target-cpu] option. You can also control
whether Rust (or LLVM) will include instructions that target optional hardware
features, e.g. hardware floating point, or vector maths operations, using [`-C
target-feature`][target-feature].

It is important to note that selecting a *target-cpu* will typically enable
*all* the optional features available from Arm on that model of CPU and your
particular implementation of that CPU may not have those features available. In
that case, you can use `-C target-feature=-option` to turn off the specific CPU
features you do not have available, leaving you with the optimised instruction
scheduling and support for the features you do have. More details are available
in the detailed target-specific documentation.

**Note:** Many target-features are currently unstable and subject to change, and
if you use them you should disassemble the compiler output and manually inspect
it to ensure only appropriate instructions for your CPU have been generated.

If you wish to use the *target-cpu* and *target-feature* options, you can add
them to your `.cargo/config.toml` file alongside any other flags your project
uses (likely linker related ones):

```toml
rustflags = [
  # Usual Arm bare-metal linker setup
  "-C", "link-arg=-Tlink.x",
  "-C", "link-arg=--nmagic",
  # tell Rust we have a Cortex-M55
  "-C", "target-cpu=cortex-m55",
  # tell Rust our Cortex-M55 doesn't have Floating-Point M-Profile Vector
  # Extensions (but it does have everything else a Cortex-M55 could have).
  "-C", "target-feature=-mve.fp"
]

[build]
target = "thumbv8m.main-none-eabihf"
```

[target-cpu]: https://doc.rust-lang.org/rustc/codegen-options/index.html#target-cpu
[target-feature]: https://doc.rust-lang.org/rustc/codegen-options/index.html#target-feature

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

Targets named `thumb*` instead of `arm*` generate Thumb (T32) code by default
instead of Arm (A32) code. Most Arm chips support both Thumb mode and Arm mode,
except that M-profile processors (`thumbv*m*-*` targets) only support Thumb-mode.

For the `arm*` targets, Thumb-mode code generation can be enabled by using `-C
target-feature=+thumb-mode`. Using the unstable
`#![feature(arm_target_feature)]`, the attribute `#[target_feature(enable =
"thumb-mode")]` can be applied to individual `unsafe` functions to cause those
functions to be compiled to Thumb-mode code.

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

* Floating-point operations are emulated in software unless LLVM is told to
  enable FPU support (either by using an `eabihf` target, specifying a
  `target-cpu` with FPU support, or using a `target-feature` to support for a
  specific kind of FPU)
* Integer division is also emulated in software on some targets, depending on
  the target, `target-cpu` and `target-feature`s.
* Older Arm architectures (e.g. Armv4, Armv5TE and Armv6-M) are limited to basic
  [`load`][atomic-load] and [`store`][atomic-store] operations, and not more
  advanced operations like [`fetch_add`][fetch-add] or
  [`compare_exchange`][compare-exchange].

`alloc` is also supported, as long as you provide your own global allocator.

Rust programs are output as ELF files.

[atomic-load]: https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicU32.html#method.load
[atomic-store]: https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicU32.html#method.store
[fetch-add]: https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicU32.html#method.fetch_add
[compare-exchange]: https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicU32.html#method.compare_exchange

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
