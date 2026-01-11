# m68k-unknown-none-elf

**Tier: 3**

Bare metal Motorola 680x0

## Target Maintainers

[@knickish](https://github.com/knickish)

## Requirements

This target requires an m68k build environment for cross-compilation which
is available on Debian, Debian-based systems, openSUSE, and other distributions.
The gnu linker is currently required, as `lld` has no support for the `m68k` architecture

On Debian-based systems, it should be sufficient to install a g++ cross-compiler for the m68k
architecture which will automatically pull in additional dependencies such as
the glibc cross development package:

```sh
apt install g++-m68k-linux-gnu
```

Binaries can be run using QEMU user emulation. On Debian-based systems, it should be
sufficient to install the package `qemu-user-static` to be able to run simple static
binaries:

```text
# apt install qemu-user-static
```

## Building

At least llvm version `19.1.5` is required to build `core` and `alloc` for this target.

## Cross-compilation

This target can be cross-compiled from a standard Debian or Debian-based, openSUSE or any
other distribution which has a basic m68k cross-toolchain available.

## Testing

Currently there is no support to run the rustc test suite for this target.

## Building Rust programs

Recommended `.cargo/config.toml`:
```toml
[unstable]
build-std = ["panic_abort", "core", "alloc"]

[target.m68k-unknown-none-elf]
# as we're building for ELF, the m68k-linux linker should be adequate
linker = "m68k-linux-gnu-ld"

# the mold linker also supports m68k, remove the above line and uncomment the
# following ones to use that instead
# linker = "clang"
# rustflags = ["-C", "link-arg=-fuse-ld=/path/to/mold/binary"]
```

Rust programs can be built for this target using:

```sh
cargo build --target m68k-unknown-none-elf
```

Very simple programs can be run using the `qemu-m68k-static` program:

```sh
qemu-m68k-static your-code
```

For more complex applications, a native (or emulated) m68k system is required for testing.
