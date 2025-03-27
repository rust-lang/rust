# `*-nuttx-elf`

**Tier: 3**

Targets for the [Apache NuttX](https://github.com/apache/nuttx).

Apache NuttX is a real-time operating system (RTOS) with an emphasis on standards compliance and small footprint. It is scalable from 8-bit to 64-bit microcontroller environments. The primary governing standards in NuttX are POSIX and ANSI standards.

NuttX adopts additional standard APIs from Unix and other common RTOSs, such as VxWorks. These APIs are used for functionality not available under the POSIX and ANSI standards. However, some APIs, like fork(), are not appropriate for deeply-embedded environments and are not implemented in NuttX.

For brevity, many parts of the documentation will refer to Apache NuttX as simply NuttX.

## Target maintainers

[@no1wudi](https://github.com/no1wudi)

## Requirements

The target name follow this format: `ARCH[-VENDOR]-nuttx-ABI`, where `ARCH` is the target architecture, `VENDOR` is the vendor name, and `ABI` is the ABI used.

The following target names are defined:

- `aarch64-unknown-nuttx`
- `armv7a-nuttx-eabi`
- `armv7a-nuttx-eabihf`
- `thumbv6m-nuttx-eabi`
- `thumbv7a-nuttx-eabi`
- `thumbv7a-nuttx-eabihf`
- `thumbv7m-nuttx-eabi`
- `thumbv7em-nuttx-eabi`
- `thumbv7em-nuttx-eabihf`
- `thumbv8m.base-nuttx-eabi`
- `thumbv8m.main-nuttx-eabi`
- `thumbv8m.main-nuttx-eabihf`
- `riscv32imc-unknown-nuttx-elf`
- `riscv32imac-unknown-nuttx-elf`
- `riscv32imafc-unknown-nuttx-elf`
- `riscv64imac-unknown-nuttx-elf`
- `riscv64gc-unknown-nuttx-elf`

## Building the target

The target can be built by enabling it in the `rustc` build:

```toml
[build]
target = "riscv32imc-unknown-nuttx-elf"

[target.'riscv32imc-unknown-nuttx-elf']
linker = "riscv-none-elf-gcc"
```

The toolchain for the target can be found in [NuttX's quick start guide](https://nuttx.apache.org/docs/latest/quickstart/install.html).


## Testing

This is a cross-compiled `no-std` target, which must be run either in a simulator
or by programming them onto suitable hardware. It is not possible to run the
Rust test-suite on this target.

## Cross-compilation toolchains and C code

This target supports C code. If interlinking with C or C++, you may need to use
`riscv-none-elf-gcc` or `arm-none-eabi-gcc` as a linker instead of `rust-lld`.
