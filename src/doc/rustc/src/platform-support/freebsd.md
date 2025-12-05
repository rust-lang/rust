# \*-unknown-freebsd

**Tier: 2/3**

[FreeBSD] multi-platform 4.4BSD-based UNIX-like operating system.

## Target maintainers

[@asomers](https://github.com/asomers)
[@MikaelUrankar](https://github.com/MikaelUrankar)

## Requirements

The `x86_64-unknown-freebsd` target is Tier 2 with host tools.
`i686-unknown-freebsd` is Tier 2 without host tools.  Other targets are Tier 3.
See [platform-support.md](../platform-support.md) for the full list.

We commit that rustc will run on all currently supported releases of
[FreeBSD][supported-releases] .  EoL releases may be supported for a time, too.
The same guarantees apply for the standard library and the libc crate.

Specific release support matrix, as of Rust 1.82.0:

| FreeBSD Release | rustc    | std      | libc    |
| --------------- | -------- | -------- | ------- |
| 10              | < 1.78.0 | ?        | ?       |
| 11              | < 1.78.0 | < 1.78.0 | current |
| 12+             | current  | current  | current |

`extern "C"` uses the official calling convention of the respective
architectures.

FreeBSD OS binaries use the ELF file format.

## Building Rust programs

The `x86_64-unknown-freebsd` and `i686-unknown-freebsd` artifacts are
distributed by the rust project and may be installed with rustup.  Other
targets are built by the ports system and may be installed with
[pkg(7)][pkg] or [ports(7)][ports].

By default the `i686-unknown-freebsd` target uses SSE2 instructions.  To build
code that does not require SSE2, build lang/rust from [ports][ports] and
disable the `SSE2` option at build time.  That will produce non-compliant
behavior.  See [issue #114479][x86-32-float-issue].

## Testing

The Rust test suite can be run natively. It can also be run from the FreeBSD
ports tree with the `make test` command from within the lang/rust directory.

[FreeBSD]: https://www.FreeBSD.org/
[supported-releases]: https://www.freebsd.org/security/#sup
[ports]: https://man.freebsd.org/cgi/man.cgi?query=ports
[pkg]: https://man.freebsd.org/cgi/man.cgi?query=pkg
[x86-32-float-issue]: https://github.com/rust-lang/rust/issues/114479
