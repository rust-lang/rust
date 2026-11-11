# powerpc64-unknown-linux-musl

**Tier: 2**

Target for 64-bit big endian PowerPC Linux programs using musl libc.
This target uses the ELF v2 ABI.

The baseline CPU required is a PowerPC 970, which means AltiVec is required and
the oldest IBM server CPU supported is therefore POWER6.

## Target maintainers

[@Gelbpunkt](https://github.com/Gelbpunkt)
[@famfo](https://github.com/famfo)
[@neuschaefer](https://github.com/neuschaefer)

## Requirements

Building the target itself requires a 64-bit big endian PowerPC compiler that
is supported by `cc-rs`.

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["powerpc64-unknown-linux-musl"]
```

Make sure your C compiler is included in `$PATH`, then add it to the
`bootstrap.toml`:

```toml
[target.powerpc64-unknown-linux-musl]
cc = "powerpc64-linux-musl-gcc"
cxx = "powerpc64-linux-musl-g++"
ar = "powerpc64-linux-musl-ar"
linker = "powerpc64-linux-musl-gcc"
```

## Building Rust programs

This target is distributed through `rustup`, and otherwise requires no
special configuration.

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

This target can be tested as normal with `x.py` on a 64-bit big endian PowerPC
host or via QEMU emulation.
