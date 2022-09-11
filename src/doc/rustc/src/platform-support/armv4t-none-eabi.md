# armv4t-none-eabi

Tier 3

Bare-metal target for any cpu in the ARMv4T architecture family, supporting
ARM/Thumb code interworking (aka `a32`/`t32`), with ARM code as the default code
generation.

In particular this supports the Gameboy Advance (GBA), but there's nothing GBA
specific with this target, so any ARMv4T device should work fine.

## Target Maintainers

* [@Lokathor](https://github.com/lokathor)

## Requirements

The target is cross-compiled, and uses static linking.

The linker that comes with rustc cannot link for this platform (the platform is
too old). You will need the `arm-none-eabi-ld` linker from a GNU Binutils
targeting ARM. This can be obtained for Windows/Mac/Linux from the [ARM
Developer Website][arm-dev], or possibly from your OS's package manager.

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

Rust programs are output as ELF files.

For running on hardware, you'll generally need to extract the "raw" program code
out of the ELF and into a file of its own. The `objcopy` program provided as
part of the GNU Binutils can do this:

```shell
arm-none-eabi-objcopy --output-target binary [in_file] [out_file]
```

## Testing

This is a cross-compiled target that you will need to emulate during testing.

Because this is a device-agnostic target, and the exact emulator that you'll
need depends on the specific device you want to run your code on.

For example, when programming for the Gameboy Advance, the
[mgba-test-runner](https://github.com/agbrs/agb) program could be used to make a
normal set of rust tests be run within the `mgba` emulator.
