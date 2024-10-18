# \*-unknown-freebsd

**Tier: 2/3**

[FreeBSD] multi-platform 4.4BSD-based UNIX-like operating system.

[FreeBSD]: https://www.FreeBSD.org/

## Target maintainers

- Alan Somers `asomers@FreeBSD.org`, https://github.com/asomers
- Mikael Urankar `mikael@FreeBSD.org`, https://github.com/MikaelUrankar

## Requirements

The `x86_64-unknown-freebsd` target is Tier 2 with host tools.
`i686-unknown-freebsd` is Tier 2 without host tools.  Other targets are Tier 3.
See [platform-support.md](../platform-support.md) for the full list.

On all architectures, rustc requires FreeBSD version 12 or later to run and
produces binaries that require FreeBSD version 12, too.  Prior to Rust 1.78.0,
rustc would run on FreeBSD 10 or later, but build binaries that required
FreeBSD 11.  libc requires FreeBSD 11 or later.

`extern "C"` uses the official calling convention of the respective architectures.

FreeBSD OS binaries use the ELF file format.

## Building Rust programs

The `x86_64-unknown-freebsd` artifacts are distributed by the rust project and
may be installed with rustup.  Other targets are built by the ports system and
may be installed with [pkg(7)](https://man.freebsd.org/cgi/man.cgi?query=pkg)
or [ports(7)](https://man.freebsd.org/cgi/man.cgi?query=ports).

## Testing

The Rust test suite can be run natively. It can also be run from the ports tree
with the `make test` command from within the lang/rust directory.
