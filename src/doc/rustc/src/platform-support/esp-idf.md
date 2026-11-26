# `*-espidf`

**Tier: 3**

Targets for the [ESP-IDF](https://github.com/espressif/esp-idf) development framework running on RISC-V and Xtensa CPUs.

ESP-IDF itself is built by the [`esp-idf-sys`](https://github.com/esp-rs/esp-idf/tree/master/esp-idf-sys) crate, as part of the `cargo` build of a Rust program. This page describes only what is specific to the Rust targets. The [esp-rs/esp-idf](https://github.com/esp-rs/esp-idf) repository is the main reference for setting up and configuring such a build.

## Target maintainers

[@ivmarkov](https://github.com/ivmarkov)
[@MabezDev](https://github.com/MabezDev)
[@SergioGasquez](https://github.com/SergioGasquez)

## Requirements

The targets support `std`. They are only used for cross-compilation, and there are no host tools for them. Binaries are ELF files, which are converted into a flash image before they are written to the chip.

The RISC-V targets are named after the instruction set of the CPU. The Xtensa targets are named after the chip, because each Xtensa chip has its own configuration of the CPU core. For all targets, `target_os` is `espidf`, `target_env` is `newlib` (also when ESP-IDF is built with picolibc, see [libc configuration](#libc-configuration)) and `target_vendor` is `espressif`. The following targets are currently defined:

| Target name               | Chip                                                              |
| ------------------------- | ----------------------------------------------------------------- |
| `riscv32imc-esp-espidf`   | [ESP32-C2](https://www.espressif.com/en/products/socs/esp32-c2)   |
| `riscv32imc-esp-espidf`   | [ESP32-C3](https://www.espressif.com/en/products/socs/esp32-c3)   |
| `riscv32imac-esp-espidf`  | [ESP32-C5](https://www.espressif.com/en/products/socs/esp32-c5)   |
| `riscv32imac-esp-espidf`  | [ESP32-C6](https://www.espressif.com/en/products/socs/esp32-c6)   |
| `riscv32imac-esp-espidf`  | [ESP32-C61](https://www.espressif.com/en/products/socs/esp32-c61) |
| `riscv32imac-esp-espidf`  | [ESP32-H2](https://www.espressif.com/en/products/socs/esp32-h2)   |
| `riscv32imac-esp-espidf`  | [ESP32-H21](https://www.espressif.com/en/products/socs/esp32-h21) |
| `riscv32imafc-esp-espidf` | [ESP32-H4](https://www.espressif.com/en/products/socs/esp32-h4)   |
| `riscv32imafc-esp-espidf` | [ESP32-P4](https://www.espressif.com/en/products/socs/esp32-p4)   |
| `riscv32imafc-esp-espidf` | [ESP32-S31](https://www.espressif.com/en/products/socs/esp32-s31) |
| `xtensa-esp32-espidf`     | [ESP32](https://www.espressif.com/en/products/socs/esp32)         |
| `xtensa-esp32s2-espidf`   | [ESP32-S2](https://www.espressif.com/en/products/socs/esp32-s2)   |
| `xtensa-esp32s3-espidf`   | [ESP32-S3](https://www.espressif.com/en/products/socs/esp32-s3)   |

When a target covers several chips, the chip is selected with the [`MCU` option][esp-idf-sys-mcu] of `esp-idf-sys`.

Not every ESP-IDF release supports every chip. See the [ESP-IDF Release and SoC Compatibility](https://github.com/espressif/esp-idf#esp-idf-release-and-soc-compatibility) table.

### libc configuration

ESP-IDF can be configured in ways that change the C types and constants which the `libc` crate, and therefore `std`, has to use. Two `--cfg` flags select them, and they must match the configuration of ESP-IDF:

- `--cfg espidf_picolibc` must be set when ESP-IDF is built with picolibc instead of newlib (`CONFIG_LIBC_PICOLIBC=y`). Picolibc is the default since ESP-IDF v6.0, and an option since ESP-IDF v5.5.
- `--cfg espidf_time32` must be set for ESP-IDF v4.x, which uses a 32-bit `time_t`. ESP-IDF v5.0 and later use a 64-bit `time_t`, which is the default. Note that at the time of writing, ESP-IDF v5.2 and older are **no longer supported by Espressif** (see the [support periods](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/versions.html#support-periods) of ESP-IDF).

The flags must be passed in the `rustflags` (or in `RUSTFLAGS`), so that `std` is built with them as well. `esp-idf-sys` checks during the build that they match the ESP-IDF it builds.

## Building the target

Rust does not ship a pre-compiled `std` for these targets. It is built from source with the unstable [`build-std`](https://doc.rust-lang.org/cargo/reference/unstable.html#build-std) feature of Cargo, as part of building a program (see below).

Building `std` alone does not need ESP-IDF. This can be used to check that `std` compiles for a target, for example:

```sh
cargo new --lib check-std
cd check-std
cargo +nightly build -Zbuild-std=std,panic_abort --target riscv32imc-esp-espidf
```

Linking a binary needs ESP-IDF, which `esp-idf-sys` builds.

## Building Rust programs

### Toolchain

The `build-std` feature needs a nightly toolchain with the `rust-src` component:

- For the RISC-V targets, use upstream nightly Rust, for example installed with `rustup toolchain install nightly --component rust-src`. For these targets, `build-std` is the only reason why a nightly toolchain is needed.
- For the Xtensa targets, use the `esp` toolchain, which is installed with [`espup`](https://github.com/esp-rs/espup). The Xtensa targets are defined in upstream `rustc`, but upstream LLVM does not yet fully support the CPU cores of these chips. The `esp` toolchain is built with Espressif's fork of LLVM instead. See also the [RISC-V and Xtensa Targets section of The Rust on ESP Book](https://docs.espressif.com/projects/rust/book/getting-started/toolchain.html).

### Other prerequisites

- [`ldproxy`](https://crates.io/crates/ldproxy), which can be installed with `cargo install ldproxy`. It passes the linker call on to the GCC toolchain that `esp-idf-sys` downloads for ESP-IDF.
- The host tools that the ESP-IDF build needs (Python, CMake, Ninja and a few others). See the [prerequisites of esp-idf-template](https://github.com/esp-rs/esp-idf-template#prerequisites).

### Configuration

A minimal `.cargo/config.toml` for a binary crate looks like this:

```toml
[build]
# Change this to the target for your chip
target = "riscv32imc-esp-espidf"

[target.'cfg(target_os = "espidf")']
linker = "ldproxy"
# With picolibc (the default since ESP-IDF v6.0):
# rustflags = ["--cfg", "espidf_picolibc"]
# With ESP-IDF v4.x:
# rustflags = ["--cfg", "espidf_time32"]

[unstable]
build-std = ["std", "panic_abort"]
```

The binary crate also needs a dependency on `esp-idf-sys` (directly, or through `esp-idf-hal` or `esp-idf-svc`), and a build script that passes the linker arguments of `esp-idf-sys` on to `ldproxy`. The ESP-IDF version, the chip and the other settings of the ESP-IDF build are options of `esp-idf-sys`, which are described in its [build options documentation][esp-idf-sys-build-options].

The easiest way to get a working setup is to generate a new project from [esp-idf-template](https://github.com/esp-rs/esp-idf-template):

```sh
cargo generate esp-rs/esp-idf-template cargo
```

## Testing

Binaries run on the chips. They are usually written to the chip and monitored with [`espflash`](https://github.com/esp-rs/espflash), for example with `espflash flash --monitor <path to the ELF file>`.

Espressif's [fork of QEMU](https://github.com/espressif/esp-toolchain-docs/blob/main/qemu/README.md) can emulate the ESP32, the ESP32-C3 and the ESP32-S3.

There is no standard way to run `cargo test` or the Rust test suite on these targets.

## Cross-compilation toolchains and C code

`esp-idf-sys` downloads the GCC toolchain that the selected ESP-IDF version uses, and builds ESP-IDF with it. Additional C code can be built in the same way, as [extra ESP-IDF components][esp-idf-sys-extra-components].

It is also possible to do the opposite: to build the Rust code as a static library and link it into a C project that is built with the CMake tooling of ESP-IDF. See the [CMake variant of esp-idf-template](https://github.com/esp-rs/esp-idf-template/blob/master/README-cmake.md).

Crates that depend on `esp-idf-sys` can also see the ESP-IDF version, the chip and the enabled ESP-IDF options as `cfg`s. See the [conditional compilation][esp-idf-sys-cfgs] section of its documentation.

[esp-idf-sys-build-options]: https://github.com/esp-rs/esp-idf/blob/master/esp-idf-sys/BUILD-OPTIONS.md
[esp-idf-sys-mcu]: https://github.com/esp-rs/esp-idf/blob/master/esp-idf-sys/BUILD-OPTIONS.md#mcu-mcu
[esp-idf-sys-extra-components]: https://github.com/esp-rs/esp-idf/blob/master/esp-idf-sys/BUILD-OPTIONS.md#extra-esp-idf-components
[esp-idf-sys-cfgs]: https://github.com/esp-rs/esp-idf/blob/master/esp-idf-sys/BUILD-OPTIONS.md#conditional-compilation
