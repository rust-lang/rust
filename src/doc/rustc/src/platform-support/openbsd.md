# \*-unknown-openbsd

**Tier: 3**

[OpenBSD] multi-platform 4.4BSD-based UNIX-like operating system.

[OpenBSD]: https://www.openbsd.org/

The target names follow this format: `$ARCH-unknown-openbsd`, where `$ARCH` specifies the target processor architecture. The following targets are currently defined:

|          Target name           | C++ library | OpenBSD Platform |
|--------------------------------|-------------|------------------|
| `aarch64-unknown-openbsd`      | libc++      | [64-bit ARM systems](https://www.openbsd.org/arm64.html)  |
| `i686-unknown-openbsd`         | libc++      | [Standard PC and clones based on the Intel i386 architecture and compatible processors](https://www.openbsd.org/i386.html) |
| `powerpc64-unknown-openbsd`    | libc++      | [IBM POWER-based PowerNV systems](https://www.openbsd.org/powerpc64.html) |
| `riscv64gc-unknown-openbsd`    | libc++      | [64-bit RISC-V systems](https://www.openbsd.org/riscv64.html) |
| `sparc64-unknown-openbsd`      | estdc++     | [Sun UltraSPARC and Fujitsu SPARC64 systems](https://www.openbsd.org/sparc64.html) |
| `x86_64-unknown-openbsd`       | libc++      | [AMD64-based systems](https://www.openbsd.org/amd64.html) |

Note that all OS versions are *major* even if using X.Y notation (`6.8` and `6.9` are different major versions) and could be binary incompatibles (with breaking changes).


## Target Maintainers

[@semarie](https://github.com/semarie)

Further contacts:

- [lang/rust](https://cvsweb.openbsd.org/cgi-bin/cvsweb/ports/lang/rust/Makefile?rev=HEAD&content-type=text/x-cvsweb-markup) maintainer (see MAINTAINER variable)

Fallback to ports@openbsd.org, OpenBSD third parties public mailing-list (with openbsd developers readers)


## Requirements

These targets are natively compiled and could be cross-compiled.
C compiler toolchain is required for the purpose of building Rust and functional binaries.

## Building

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["$ARCH-unknown-openbsd"]

[target.$ARCH-unknown-openbsd]
cc = "$ARCH-openbsd-cc"
```

## Cross-compilation

These targets can be cross-compiled, but LLVM might not build out-of-box.

## Testing

The Rust testsuite could be run natively.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for these targets.
