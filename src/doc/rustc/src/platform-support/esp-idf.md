# `*-esp-espidf`

**Tier: 3**

Targets for the [ESP-IDF](https://github.com/espressif/esp-idf) development framework running on RISC-V and Xtensa CPUs.

## Target maintainers

- Ivan Markov [@ivmarkov](https://github.com/ivmarkov)
- Scott Mabin [@MabezDev](https://github.com/MabezDev)

## Requirements

The target names follow this format: `$ARCH-esp-espidf`, where `$ARCH` specifies the target processor architecture. The following targets are currently defined:

|          Target name           | Target CPU(s)         |
|--------------------------------|-----------------------|
| `riscv32imc-esp-espidf`        |  [ESP32-C3](https://www.espressif.com/en/products/socs/esp32-c3)             |

The minimum supported ESP-IDF version is `v4.3`, though it is recommended to use the latest stable release if possible.

## Building the target

The target can be built by enabling it for a `rustc` build. The `build-std` feature is required to build the standard library for ESP-IDF. `ldproxy` is also required for linking, it can be installed from crates.io.

```toml
[build]
target = ["$ARCH-esp-espidf"]

[target.$ARCH-esp-espidf]
linker = "ldproxy"

[unstable]
build-std = ["std", "panic_abort"]
```

The `esp-idf-sys` crate will handle the compilation of ESP-IDF, including downloading the relevant toolchains for the build.

## Cross-compilation toolchains and C code

`esp-idf-sys` exposes the toolchain used in the compilation of ESP-IDF, see the crate [documentation for build output propagation](https://github.com/esp-rs/esp-idf-sys#conditional-compilation) for more information.
