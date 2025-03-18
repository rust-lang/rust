# riscv64gc-unknown-linux-musl

**Tier: 2**

Target for RISC-V Linux programs using musl libc.

## Target maintainers

- [@Amanieu](https://github.com/Amanieu)
- [@kraj](https://github.com/kraj)

## Requirements

Building the target itself requires a RISC-V compiler that is supported by `cc-rs`.

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["riscv64gc-unknown-linux-musl"]
```

Make sure your C compiler is included in `$PATH`, then add it to the `bootstrap.toml`:

```toml
[target.riscv64gc-unknown-linux-musl]
cc = "riscv64-linux-gnu-gcc"
cxx = "riscv64-linux-gnu-g++"
ar = "riscv64-linux-gnu-ar"
linker = "riscv64-linux-gnu-gcc"
```

## Building Rust programs

This target are distributed through `rustup`, and otherwise require no
special configuration.

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

This target can be tested as normal with `x.py` on a RISC-V host or via QEMU
emulation.
