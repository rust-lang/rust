# `*-unknown-serenity`

**Tier: 3**

SerenityOS is a graphical Unix-like operating system for 64-bit x86, Arm, and RISC-V computers.

## Target maintainers

[@ADKaster](https://github.com/ADKaster)
[@nico](https://github.com/nico)

## Requirements

Host tools are not supported. `std` is not yet supported. There are no releases, as
SerenityOS is a hobbyist OS.

A checkout of [`serenity`](https://github.com/SerenityOS/serenity) is required to build
the operating system image. The simplest cross-compilation environment is the
serenity [Ports tree](https://github.com/SerenityOS/serenity/tree/master/Ports#readme).
Out-of-tree cross-compilation has been shown to work, but is not a typical developer
workflow.

Target binaries are in the ELF format and are expected to call into the Kernel
via libc. Direct syscalls are not supported due to an [`msyscall` mitigation]
(https://man.serenityos.org/man7/Mitigations.html#syscall-call-from-verification).

## Building the target

The targets can be built by enabling them for a `rustc` build, for example:

```toml
[build]
build-stage = 1
target = ["aarch64-unknown-serenity", "x86_64-unknown-serenity"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Target binaries can be included in a default developer QEMU image by copying
the binary and any required runtime dependencies into a subfolder under
`Build/<arch>/Root` or `Build/<arch>clang/Root`, depending on the cross-compilation
toolchain used when building.

Alternatively, a [`sync-local.sh`](https://github.com/SerenityOS/serenity/blob/master/Documentation/AdvancedBuildInstructions.md#customizing-the-disk-image)
script can be used to modify the disk image during an image build.

Running the rust test suite is not yet supported.

## Cross-compilation toolchains and C code

The operating system is written in C++, so a C/C++ compiler is required.
The target triples are as expected, `$ARCH-unknown-serenity`.
