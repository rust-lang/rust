# `armv5te-none-eabi`

**Tier: 3**

Bare-metal target for any cpu in the ARMv5TE architecture family, supporting
ARM/Thumb code interworking (aka `a32`/`t32`), with `a32` code as the default code
generation.

The `thumbv5te-none-eabi` target is the same as this one, but the instruction set defaults to `t32`.

## Target Maintainers

* [@QuinnPainter](https://github.com/QuinnPainter)

## Requirements

The target is cross-compiled, and uses static linking.

By default, the `lld` linker included with Rust will be used.

However, you may want to use the `arm-none-eabi-ld` linker instead. This can be obtained for Windows/Mac/Linux from the [ARM
Developer Website][arm-dev], or possibly from your OS's package manager. To use it, add the following to your `.cargo/config.toml`:

```toml
[target.armv5te-none-eabi]
linker = "arm-none-eabi-ld"
```

[arm-dev]: https://developer.arm.com/Tools%20and%20Software/GNU%20Toolchain

This target doesn't provide a linker script, you'll need to bring your own
according to the specific device you want to target. Pass
`-Clink-arg=-Tyour_script.ld` as a rustc argument to make the linker use
`your_script.ld` during linking.

## Building Rust Programs

Because it is Tier 3, rust does not yet ship pre-compiled artifacts for this target.

Just use the `build-std` nightly cargo feature to build the `core` library. You
can pass this as a command line argument to cargo, or your `.cargo/config.toml`
file might include the following lines:

```toml
[unstable]
build-std = ["core"]
```

Most of `core` should work as expected, with the following notes:
* the target is "soft float", so `f32` and `f64` operations are emulated in
  software.
* integer division is also emulated in software.
* the target is old enough that it doesn't have atomic instructions.

`alloc` is also supported, as long as you provide your own global allocator.

Rust programs are output as ELF files.

## Testing

This is a cross-compiled target that you will need to emulate during testing.

Because this is a device-agnostic target, and the exact emulator that you'll
need depends on the specific device you want to run your code on.

For example, when programming for the DS, you can use one of the several available DS emulators, such as [melonDS](https://melonds.kuribo64.net/).
