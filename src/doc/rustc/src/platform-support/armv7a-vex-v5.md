# `armv7a-vex-v5`

**Tier: 3**

Allows compiling user programs for the [VEX V5 Brain](https://www.vexrobotics.com/276-4810.html), a microcontroller for educational and competitive robotics.

Rust support for this target is not affiliated with VEX Robotics or IFI.

## Target maintainers

This target is maintained by members of the [vexide](https://github.com/vexide) organization:

- [@lewisfm](https://github.com/lewisfm)
- [@Tropix126](https://github.com/Tropix126)
- [@Gavin-Niederman](https://github.com/Gavin-Niederman)
- [@max-niederman](https://github.com/max-niederman)

## Requirements

This target is cross-compiled and currently requires `#![no_std]`. Dynamic linking is unsupported.

When compiling for this target, the "C" calling convention maps to AAPCS with VFP registers (hard float ABI) and the "system" calling convention maps to AAPCS without VFP registers (soft float ABI).

This target generates binaries in the ELF format that may uploaded to the brain with external tools.

## Building the target

You can build Rust with support for this target by adding it to the `target` list in `bootstrap.toml`, and then running `./x build --target armv7a-vex-v5 compiler`.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

When the compiler builds a binary, an ELF build artifact will be produced. Additional tools are required for this artifact to be recognizable to VEXos as a user program.

The [cargo-v5](https://github.com/vexide/cargo-v5) tool is capable of creating binaries that can be uploaded to the V5 brain. This tool wraps the `cargo build` command by supplying arguments necessary to build the target and produce an artifact recognizable to VEXos, while also providing functionality for uploading over USB to a V5 Controller or Brain.

To install the tool, run:

```sh
cargo install cargo-v5
```

The following fields in your project's `Cargo.toml` are read by `cargo-v5` to configure upload behavior:

```toml
[package.metadata.v5]
# Slot number to upload the user program to. This should be from 1-8.
slot = 1
# Program icon/thumbnail that will be displayed on the dashboard.
icon = "cool-x"
# Use gzip compression when uploading binaries.
compress = true
```

To build an uploadable BIN file using the release profile, run:

```sh
cargo v5 build --release
```

Programs can also be directly uploaded to the brain over a USB connection immediately after building:

```sh
cargo v5 upload --release
```

## Testing

Binaries built for this target can be run in an emulator (such as [vex-v5-qemu](https://github.com/vexide/vex-v5-qemu)), or uploaded to a physical device over a serial (USB) connection.

The default Rust test runner is not supported.

The Rust test suite for `library/std` is not yet supported.

## Cross-compilation toolchains and C code

This target can be cross-compiled from any host.

Linking to C libraries is not supported.
