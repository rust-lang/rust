# powerpc64-unknown-linux-gnuelfv2

**Tier: 3**

Target for 64-bit big endian PowerPC Linux programs using the ELFv2 ABI and
the GNU C library.

## Target maintainers

[@Gelbpunkt](https://github.com/Gelbpunkt)

## Requirements

Building the target itself requires a 64-bit big endian PowerPC compiler that
uses the ELFv2 ABI and is supported by `cc-rs`.

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["powerpc64-unknown-linux-gnuelfv2"]
```

Make sure your C compiler is included in `$PATH`, then add it to the
`bootstrap.toml`:

```toml
[target.powerpc64-unknown-linux-gnuelfv2]
cc = "powerpc64-linux-gnu-gcc"
cxx = "powerpc64-linux-gnu-g++"
ar = "powerpc64-linux-gnu-ar"
linker = "powerpc64-linux-gnu-gcc"
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will first need to build Rust with the target enabled (see
"Building the target" above).

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

This target can be tested as normal with `x.py` on a 64-bit big endian PowerPC
host or via QEMU emulation.
