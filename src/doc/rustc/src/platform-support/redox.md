# `*-unknown-redox`

**Tier: 2/3**

Targets for the [Redox OS](https://redox-os.org/) operating
system.

Target triplets available so far:

- `x86_64-unknown-redox` (tier 2)
- `aarch64-unknown-redox` (tier 3)
- `i586-unknown-redox` (tier 3)

## Target maintainers

- Jeremy Soller ([@jackpot51](https://github.com/jackpot51))

## Requirements

These targets are natively compiled and can be cross-compiled. Std is fully supported.

The targets are only expected to work with the latest version of Redox OS as the ABI is not yet stable.

`extern "C"` uses the official calling convention of the respective architectures.

Redox OS binaries use ELF as file format.

## Building the target

You can build Rust with support for the targets by adding it to the `target` list in `bootstrap.toml`. In addition a copy of [relibc] needs to be present in the linker search path.

```toml
[build]
build-stage = 1
target = [
    "<HOST_TARGET>",
    "x86_64-unknown-redox",
    "aarch64-unknown-redox",
    "i586-unknown-redox",
]
```

[relibc]: https://gitlab.redox-os.org/redox-os/relibc

## Building Rust programs and testing

Rust does not yet ship pre-compiled artifacts for Redox OS except for x86_64-unknown-redox.

The easiest way to build and test programs for Redox OS is using [redoxer](https://gitlab.redox-os.org/redox-os/redoxer) which sets up the required compiler toolchain for building as well as runs programs inside a Redox OS VM using QEMU.

## Cross-compilation toolchains and C code

The target supports C code. Pre-compiled C toolchains can be found at <https://static.redox-os.org/toolchain/>.
