# riscv64gc-unknown-linux-musl

**Tier: 2 (with Host Tools)**

Linux musl libc RISC-V target using the *RV64I* base instruction set with the
*G* collection of extensions, as well as the *C* extension.


## Target maintainers

[@Amanieu](https://github.com/Amanieu)
[@kraj](https://github.com/kraj)

## Requirements

This target requires:

* Linux Kernel version 4.20 or later
* musl libc 1.2.5 or later


## Building the target

This target is distributed through `rustup`, and otherwise requires no
special configuration.

If you need to build your own Rust then the targets can be enabled in
`bootstrap.toml`. For example:

```toml
[build]
target = ["riscv64gc-unknown-linux-musl"]
```


## Building Rust programs


On a riscv64gc-unknown-linux-musl host, the `riscv64gc-unknown-linux-musl`
target should be automatically installed and used by default.

On all other hosts, add the target:

```bash
rustup target add riscv64gc-unknown-linux-musl
```

Then cross compile crates with:

```bash
cargo build --target riscv64gc-unknown-linux-musl
```

## Testing

The target can be tested as normal with `x.py` on a RISC-V host or via QEMU
emulation.
