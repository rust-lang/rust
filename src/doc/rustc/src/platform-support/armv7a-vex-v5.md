# `armv7a-vex-v5`

**Tier: 3**


Allows compiling user programs for the [VEX V5 Brain](https://www.vexrobotics.com/276-4810.html), a microcontroller for educational and competitive robotics.

Rust support for this target is not affiliated with VEX Robotics or IFI.

## Target maintainers

This target is maintained by members of the [vexide](https://github.com/vexide) organization:

- [@Tropix126](https://github.com/Tropix126)
- [@Gavin-Niederman](https://github.com/Gavin-Niederman)
- [@max-niederman](https://github.com/max-niederman)
- [@doinkythederp](https://github.com/doinkythederp)

## Requirements

This target is cross-compiled. Dynamic linking is unsupported.

`#![no_std]` crates can be built using `build-std` to build `core` and optionally
`alloc`. Unwinding panics are not yet supported.

`std` is partially implemented, but many modules (such as `thread`, `process`, `net`, etc...) will return errors. An allocator is provided along with partial support for the `time`, `env` and `io` modules. Filesystem operations over SDCard through `std::fs` are partially supported within the restrictions of the user enviornment (e.g. directories cannot be created, filesystem objects cannot be removed).

This target generates binaries in the ELF format that may uploaded to the brain with external tools.

## Building the target

Rust does not ship pre-compiled artifacts for this target. You can use the `build-std` feature to build ELF binaries with `std` support.

`.cargo/config.toml`:

```toml
[build]
target = "armv7a-vex-v5"

[unstable]
build-std = ["std", "panic_abort"]
build-std-features = ["compiler-builtins-mem"]
```

## Building Rust programs

The recommended way to build artifacts that run on V5 Brain is by using the [cargo-v5](https://github.com/vexide/cargo-v5) tool. This tool wraps the `cargo build` command by supplying arguments necessary to build the target, while also providing functionality for uploading over USB to a V5 Controller or Brain.

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

This target does not link to C libraries. OS calls are implemented in rust through the [vex-sdk](https://github.com/vexide/vex-sdk) crate. No `libc` or crt0 implementation is present on this target.