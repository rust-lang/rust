# powerpc64le-unknown-linux-musl

**Tier: 2**

Target for 64-bit little endian PowerPC Linux programs using musl libc.

## Target maintainers

- [@Gelbpunkt](https://github.com/Gelbpunkt)
- [@famfo](https://github.com/famfo)
- [@neuschaefer](https://github.com/neuschaefer)

## Requirements

Building the target itself requires a 64-bit little endian PowerPC compiler that is supported by `cc-rs`.

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["powerpc64le-unknown-linux-musl"]
```

Make sure your C compiler is included in `$PATH`, then add it to the `bootstrap.toml`:

```toml
[target.powerpc64le-unknown-linux-musl]
cc = "powerpc64le-linux-musl-gcc"
cxx = "powerpc64le-linux-musl-g++"
ar = "powerpc64le-linux-musl-ar"
linker = "powerpc64le-linux-musl-gcc"
```

## Building Rust programs

This target are distributed through `rustup`, and otherwise require no
special configuration.

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

This target can be tested as normal with `x.py` on a 64-bit little endian
PowerPC host or via QEMU emulation.
