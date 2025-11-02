# `aarch64-unknown-linux-gnu`

**Tier: 1 (with Host Tools)**

Target for 64-bit little endian ARMv8-A Linux 4.1+ programs using glibc 2.17+.

## Target maintainers

- [@rust-lang/arm-maintainers][arm_maintainers] ([rust@arm.com][arm_email])

[arm_maintainers]: https://github.com/rust-lang/team/blob/master/teams/arm-maintainers.toml
[arm_email]: mailto:rust@arm.com

## Requirements

Building the target itself requires a 64-bit little endian ARMv8-A compiler that is supported by
`cc-rs`.

## Building the target

The target can be built by enabling it for a `rustc` build:

```toml
[build]
target = ["aarch64-unknown-linux-gnu"]
```

If cross-compiling, make sure your C compiler is included in `$PATH`, then add it to the
`bootstrap.toml`:

```toml
[target.aarch64-unknown-linux-musl]
cc = "aarch64-linux-gnu-gcc"
cxx = "aarch64-linux-gnu-g++"
ar = "aarch64-linux-gnu-ar"
linker = "aarch64-linux-gnu-gcc"
```

## Building Rust programs

This target is distributed through `rustup`, and otherwise requires no special configuration.

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

This target can be tested as normal with `x.py` on a 64-bit little endian ARMv8-A host or via QEMU
emulation.
