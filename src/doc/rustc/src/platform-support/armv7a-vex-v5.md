# `armv7a-vex-v5`

**Tier: 3**

Allows compiling user programs for the [VEX V5 Brain](https://www.vexrobotics.com/276-4810.html), a microcontroller for educational and competitive robotics.

Rust support for this target is not affiliated with VEX Robotics or IFI, and does not link against any official VEX SDK.

## Target maintainers

This target is maintained by members of the [vexide](https://github.com/vexide) organization:

- [@lewisfm](https://github.com/lewisfm)
- [@Tropix126](https://github.com/Tropix126)
- [@Gavin-Niederman](https://github.com/Gavin-Niederman)
- [@max-niederman](https://github.com/max-niederman)

## Requirements

This target is cross-compiled. Dynamic linking is unsupported.

`#![no_std]` crates can be built using `build-std` to build `core` and `panic_abort` and optionally `alloc`. Unwinding panics are not yet supported on this target.

`std` has only partial support due to platform limitations. Notably:

- `std::process` and `std::net` are unimplemented. `std::thread` only supports sleeping and yielding, as this is a single-threaded environment.
- `std::time` has full support for `Instant`, but no support for `SystemTime`.
- `std::io` has full support for `stdin`/`stdout`/`stderr`. `stdout` and `stderr` both write to USB channel 1 on this platform and are not differentiated.
- `std::fs` has limited support for reading or writing to files. The following features are unsupported:
  - All directory operations (including `mkdir` and `readdir`), although reading directories is possible through [third-party crates](https://docs.rs/vex-sdk/latest/vex_sdk/file/fn.vexFileDirectoryGet.html)
  - Deleting files and directories
  - File metadata other than file size and type (that is, file vs. directory)
  - Opening files with an uncommon combination of open options, such as read + write at the same time.
    The supported modes for opening files are in read-only mode, append mode, or write mode (with or without truncation).
- A global allocator implemented on top of `dlmalloc` is provided.
- Modules that do not need to interact with the OS beyond allocation, such as `std::collections`, `std::hash`, `std::future`, `std::sync`, etc., are fully supported.
- Random number generation and hashing is insecure, as there is no reliable source of entropy on this platform.

When compiling for this target, the "C" calling convention maps to AAPCS with VFP registers (hard float ABI) and the "system" calling convention maps to AAPCS without VFP registers (softfp ABI).

This target generates binaries in the ELF format that may be uploaded to the brain with external tools.

### Platform SDKs

To use most platform-specific APIs, users must configure a supporting runtime SDK for `libstd` to link against. Official *VEXcode* SDKs from VEX can be downloaded and linked via the [`vex-sdk-vexcode`](https://crates.io/crates/vex-sdk-vexcode) crate, but they have a restrictive redistribution policy that might not be suitable for all projects. The suggested SDK for open-source projects is the community-supported [`vex-sdk-jumptable`](https://crates.io/crates/vex-sdk-jumptable) crate. SDK implementations are generally thin wrappers over system calls, so projects should not expect to see significant differences in behavior depending on which SDK they use.

Libraries may access symbols from the active VEX SDK without depending on a specific implementation by using the [`vex-sdk`](https://crates.io/crates/vex-sdk) crate.

## Building the target

You can build Rust with support for this target by adding it to the `target` list in `bootstrap.toml`, and then running `./x build --target armv7a-vex-v5 compiler`.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for this target, you will either need to build Rust with the target enabled (see "Building the target" above), or build your own copy of `std` using `build-std` or similar.

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

### Hello World program

```rs
use ::vex_sdk_jumptable as _; // Bring VEX SDK symbols into scope

fn main() {
    println!("Hello, world");
}
```

## Testing

Binaries built for this target can be run in an emulator (such as [vex-v5-qemu](https://github.com/vexide/vex-v5-qemu)), or uploaded to a physical device over a USB serial connection.

The default Rust test runner is not supported.

The Rust test suite for `library/std` is not yet supported.

## Cross-compilation toolchains and C code

This target can be cross-compiled from any host.

The recommended configuration for compiling compatible C code is to use the [Arm Toolchain for Embedded](https://github.com/arm/arm-toolchain/tree/arm-software/arm-software/embedded#readme) with the following compilation flags:

```sh
clang --target=arm-none-eabi -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -fno-pic -fno-exceptions -fno-rtti -funwind-tables
```

The following Cargo configuration can be used to link with picolibc (the libc used by the Arm Toolchain for Embedded):

```toml
[target.armv7a-vex-v5]
# We use ARM Clang as a linker because ld.lld by itself doesn't include the
# multilib logic for resolving static libraries.
linker = "clang"

rustflags = [
    # These link flags resolve to this sysroot:
    # `…/arm-none-eabi/armv7a_hard_vfpv3_d16_unaligned`
    # (hard float / VFP version 3 with 16 regs / unaligned access)
    "-Clink-arg=--target=armv7a-none-eabihf",

    # To disable crt0 and use Rust's _boot implementation
    # (or something custom):
    #"-Clink-arg=-nostartfiles",

    # Explicit `-lc` required because Rust calls the linker with
    # `-nodefaultlibs` which disables libc, libm, etc.
    "-Clink-arg=-lc",
]
```

You may also want to set these environment variables so that third-party crates use the correct C compiler:

```sh
PATH=/path/to/arm-toolchain/bin:$PATH
CC_armv7a_vex_v5=clang
AR_armv7a_vex_v5=clang
CFLAGS_armv7a_vex_v5=[See above]
```

### CMake

It may be helpful to create a CMake toolchain like the following if you are depending on the `cmake` crate:

```cmake
# toolchain.cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(triple armv7a-none-eabihf)

set(CMAKE_C_COMPILER clang)
set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_COMPILER_TARGET ${triple})
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
```

You can enable it by setting the following environment variable alongside the previously mentioned environment variables:

```sh
CMAKE_TOOLCHAIN_FILE_armv7a_vex_v5=/path/to/toolchain.cmake
```

### Implementation of libc functions

You may have to implement [certain system support functions](https://github.com/picolibc/picolibc/blob/main/doc/os.md) for some parts of libc to work properly.
