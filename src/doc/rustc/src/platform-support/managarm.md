# `*-unknown-managarm-mlibc`

**Tier: 3**

## Target Maintainers

- [@no92](https://github.com/no92)
- [@64](https://github.com/64)
- [@Dennisbonke](https://github.com/Dennisbonke)

## Requirements

This target is cross-compiled. There is currently no support for `std` yet. It generates binaries in the ELF format. Currently, we support the `x86_64`, `aarch64` and `riscv64gc` architectures. The examples below `$ARCH` should be substituted for one of the supported architectures.

## Building the target

Managarm has upstream support in LLVM since the release of 21.1.0.

Set up your `bootstrap.toml` like this:

```toml
change-id = 142379

[llvm]
targets = "X86;AArch64;RISCV"
download-ci-llvm = false

[build]
target = ["$ARCH-unknown-managarm-mlibc", "x86_64-unknown-linux-gnu"]

[target.x86_64-unknown-linux-gnu]
llvm-config = "/path/to/your/llvm/bin/llvm-config"

[target.$ARCH-unknown-managarm-mlibc]
llvm-config = "/path/to/your/llvm/bin/llvm-config"
```

## Building Rust programs

Build a `$ARCH-managarm-gcc` using our [gcc fork](https://github.com/managarm/gcc).

```toml
[build]
rustc = "/path/to/the/rust-prefix/bin/rustc"
target = "$ARCH-unknown-managarm-mlibc"

[target.$ARCH-unknown-managarm-mlibc]
linker = "/path/to/the/managarm-gcc/bin/$ARCH-managarm-gcc"
```

## Testing

This target does not support running the Rust testsuite yet.
