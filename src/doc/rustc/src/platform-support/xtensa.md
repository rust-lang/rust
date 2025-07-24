# `xtensa-*-none-elf`

**Tier: 3**

Targets for Xtensa CPUs.

## Target maintainers

[@MabezDev](https://github.com/MabezDev)
[@SergioGasquez](https://github.com/SergioGasquez)

## Requirements

The target names follow this format: `xtensa-$CPU`, where `$CPU` specifies the target chip. The following targets are currently defined:

| Target name               | Target CPU(s)                                                   |
| ------------------------- | --------------------------------------------------------------- |
| `xtensa-esp32-none-elf`   | [ESP32](https://www.espressif.com/en/products/socs/esp32)       |
| `xtensa-esp32s2-none-elf` | [ESP32-S2](https://www.espressif.com/en/products/socs/esp32-s2) |
| `xtensa-esp32s3-none-elf` | [ESP32-S3](https://www.espressif.com/en/products/socs/esp32-s3) |


Xtensa targets that support `std` are documented in the [ESP-IDF platform support document](esp-idf.md)

## Building the targets

The targets can be built by installing the [Xtensa enabled Rust channel](https://github.com/esp-rs/rust/). See instructions in the [RISC-V and Xtensa Targets section of The Rust on ESP Book](https://docs.espressif.com/projects/rust/book/installation/index.html).
