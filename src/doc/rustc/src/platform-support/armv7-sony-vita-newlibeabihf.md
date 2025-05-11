# armv7-sony-vita-newlibeabihf

**Tier: 3**

This tier supports the ARM Cortex A9 processor running on a PlayStation Vita console.

Rust support for this target is not affiliated with Sony, and is not derived
from nor used with any official Sony SDK.

## Target maintainers

[@nikarh](https://github.com/nikarh)
[@pheki](https://github.com/pheki)
[@zetanumbers](https://github.com/zetanumbers)

## Requirements

This target is cross-compiled, and requires installing [VITASDK](https://vitasdk.org/) toolchain on your system. Dynamic linking is not supported.

`#![no_std]` crates can be built using `build-std` to build `core`, and optionally
`alloc`, and `panic_abort`.

`std` is partially supported, but mostly works. Some APIs are unimplemented
and will simply return an error, such as `std::process`.

This target generates binaries in the ELF format with thumb ISA by default.

Binaries are linked with `arm-vita-eabi-gcc` provided by VITASDK toolchain.


## Building the target

Rust does not ship pre-compiled artifacts for this target. You can use `build-std` flag to build ELF binaries with `std`:

```sh
cargo build -Z build-std=std,panic_abort --target=armv7-sony-vita-newlibeabihf --release
```

## Building Rust programs

The recommended way to build artifacts that can be installed and run on PlayStation Vita is by using the [cargo-vita](https://github.com/vita-rust/cargo-vita) tool. This tool uses `build-std` and VITASDK toolchain to build artifacts runnable on Vita.

To install the tool run:

```sh
cargo install cargo-vita
```

[VITASDK](https://vitasdk.org/) toolchain must be installed, and the `VITASDK` environment variable must be set to its location, e.g.:

```sh
export VITASDK=/opt/vitasdk
```

Add the following section to your project's `Cargo.toml`:


```toml
[package.metadata.vita]
# A unique 9 character alphanumeric identifier of the app.
title_id = "RUSTAPP01"
# A title that will be used for the app. Optional, name will be used if not defined
title_name = "My application"
```

To build a VPK with ELF in the release profile, run:

```sh
cargo vita build vpk --release
```

After building a *.vpk file it can be uploaded to a PlayStation Vita and installed, or used with a [Vita3K](https://vita3k.org/) emulator.

## Testing

The default Rust test runner is supported, and tests can be compiled to an elf and packed to a *.vpk file using `cargo-vita` tool. Filtering tests is not currently supported since passing command-line arguments to the executable is not supported on Vita, so the runner will always execute all tests.

The Rust test suite for `library/std` is not yet supported.

## Cross-compilation

This target can be cross-compiled from `x86_64` on Windows, MacOS or Linux systems. Other hosts are not supported for cross-compilation.
