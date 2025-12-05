# mips64-unknown-linux-muslabi64

**Tier: 3**

Target for 64-bit big endian MIPS Linux programs using musl libc and the N64 ABI.

## Target maintainers

[@Gelbpunkt](https://github.com/Gelbpunkt)

## Requirements

Building the target itself requires a 64-bit big endian MIPS compiler that is
supported by `cc-rs`.

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["mips64-unknown-linux-muslabi64"]
```

Make sure your C compiler is included in `$PATH`, then add it to the
`bootstrap.toml`:

```toml
[target.mips64-unknown-linux-muslabi64]
cc = "mips64-linux-musl-gcc"
cxx = "mips64-linux-musl-g++"
ar = "mips64-linux-musl-ar"
linker = "mips64-linux-musl-gcc"
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will first need to build Rust with the target enabled (see
"Building the target" above).

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

This target can be tested as normal with `x.py` on a 64-bit big endian MIPS
host or via QEMU emulation.
