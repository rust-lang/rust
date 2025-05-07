# `powerpc64le-unknown-linux-gnu`

**Tier: 2**

Target for 64-bit little endian PowerPC Linux programs

## Target maintainers

[@daltenty](https://github.com/daltenty)
[@gilamn5tr](https://github.com/gilamn5tr)

## Requirements

Building the target itself requires a 64-bit little endian PowerPC compiler that is supported by `cc-rs`.

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["powerpc64le-unknown-linux-gnu"]
```

Make sure your C compiler is included in `$PATH`, then add it to the `config.toml`:

```toml
[target.powerpc64le-unknown-linux-gnu]
cc = "powerpc64le-linux-gnu-gcc"
cxx = "powerpc64le-linux-gnu-g++"
ar = "powerpc64le-linux-gnu-ar"
linker = "powerpc64le-linux-gnu-gcc"
```

## Building Rust programs

This target is distributed through `rustup`, and requires no special
configuration.

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

This target can be tested as normal with `x.py` on a 64-bit little endian
PowerPC host or via QEMU emulation.
