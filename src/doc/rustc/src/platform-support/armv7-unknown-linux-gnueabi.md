# `armv7-unknown-linux-gnueabi` and `armv7-unknown-linux-gnueabihf`

* **Tier: 2 (with Host Tools)** for `armv7-unknown-linux-gnueabihf`
* **Tier: 2** for `armv7-unknown-linux-gnueabi`

Target for 32-bit little endian ARMv7-A Linux 3.2+ programs using glibc 2.17+.

## Target maintainers

- [@rust-lang/arm-maintainers][arm_maintainers] ([rust@arm.com][arm_email])

[arm_maintainers]: https://github.com/rust-lang/team/blob/master/teams/arm-maintainers.toml
[arm_email]: mailto:rust@arm.com

## Requirements

Building the targets themselves requires a 32-bit little endian ARMv7-A compiler that is supported
by `cc-rs`.

## Building the target

These targets can be built by enabling it for a `rustc` build:

```toml
[build]
target = ["armv7-unknown-linux-gnueabihf", "armv7-unknown-linux-gnueabi"]
```

If cross-compiling, make sure your C compiler is included in `$PATH`, then add it to the
`bootstrap.toml`:

```toml
[target.aarch64-unknown-linux-musl]
cc = "arm-linux-gnu-gcc"
cxx = "arm-linux-gnu-g++"
ar = "arm-linux-gnu-ar"
linker = "arm-linux-gnu-gcc"
```

## Building Rust programs

These targets is distributed through `rustup`, and otherwise requires no special configuration.

## Cross-compilation

These targets can be cross-compiled from any host.

## Testing

These targets can be tested as normal with `x.py` on a 32-bit little endian ARMv7-A host or via
QEMU emulation.
