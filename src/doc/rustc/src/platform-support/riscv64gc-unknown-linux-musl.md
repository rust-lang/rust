# riscv64gc-unknown-linux-musl

**Tier: 2 (with Host Tools)**

Target for RISC-V Linux programs using musl libc.

## Target maintainers

[@Amanieu](https://github.com/Amanieu)
[@kraj](https://github.com/kraj)

## Requirements

Building the target itself requires a RISC-V compiler that is supported by `cc-rs`.

## Building the target

The target can be built by enabling it for a `rustc` build.

These targets are distributed through `rustup`, and otherwise require no
special configuration.

If you need to build your own Rust then the targets can be enabled in
`bootstrap.toml`. For example:

```toml
[build]
target = ["riscv64gc-unknown-linux-musl"]
```


## Building Rust programs

This target are distributed through `rustup`, and otherwise require no
special configuration.

On a RISC-V host, the `riscv64gc-unknown-linux-musl` target should be
automatically installed and used by default.

On a non-RISC-V host, add the target:

```bash
rustup target add riscv64gc-unknown-linux-musl
```

Then cross compile crates with:

```bash
cargo build --target riscv64gc-unknown-linux-musl
```


## Testing

This target can be tested as normal with `x.py` on a RISC-V host or via QEMU
emulation.
