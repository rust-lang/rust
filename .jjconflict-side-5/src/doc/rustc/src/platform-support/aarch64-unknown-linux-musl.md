# aarch64-unknown-linux-musl

**Tier: 2**

Target for 64-bit little endian ARMv8-A Linux programs using musl libc.

## Target maintainers

[@Gelbpunkt](https://github.com/Gelbpunkt)
[@famfo](https://github.com/famfo)

## Requirements

Building the target itself requires a 64-bit little endian ARMv8-A compiler
that is supported by `cc-rs`.

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["aarch64-unknown-linux-musl"]
```

Make sure your C compiler is included in `$PATH`, then add it to the
`bootstrap.toml`:

```toml
[target.aarch64-unknown-linux-musl]
cc = "aarch64-linux-musl-gcc"
cxx = "aarch64-linux-musl-g++"
ar = "aarch64-linux-musl-ar"
linker = "aarch64-linux-musl-gcc"
```

## Building Rust programs

This target is distributed through `rustup`, and otherwise requires no
special configuration.

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

This target can be tested as normal with `x.py` on a 64-bit little endian
ARMv8-A host or via QEMU emulation.
