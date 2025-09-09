# `loongarch*-unknown-linux-*`

**Tier: 2 (with Host Tools)**

[LoongArch][la-docs] Linux targets.
LoongArch is a RISC ISA developed by Loongson Technology Corporation Limited.

| Target | Description |
|--------|-------------|
| `loongarch64-unknown-linux-gnu` | LoongArch64 Linux, LP64D ABI (kernel 5.19, glibc 2.36), LSX required |
| `loongarch64-unknown-linux-musl` | LoongArch64 Linux, LP64D ABI (kernel 5.19, musl 1.2.5), LSX required |

These support both native and cross builds, and have full support for `std`.

Reference material:

* [LoongArch ISA manuals][la-docs]
* [Application Binary Interface for the LoongArch&trade; Architecture][la-abi-specs]

[la-abi-specs]: https://github.com/loongson/la-abi-specs
[la-docs]: https://loongson.github.io/LoongArch-Documentation/README-EN.html

## Target maintainers

[@heiher](https://github.com/heiher)
[@xen0n](https://github.com/xen0n)

## Requirements

### OS Version

The minimum supported Linux version is 5.19.

Some Linux distributions, mostly commercial ones, may provide forked Linux
kernels that has a version number less than 5.19 for their LoongArch ports.
Such kernels may still get patched to be compatible with the upstream Linux
5.19 UAPI, therefore supporting the targets described in this document, but
this is not always the case. The `rustup` installer contains a check for this,
and will abort if incompatibility is detected.

### Host toolchain

The targets require a reasonably up-to-date LoongArch toolchain on the host.
Currently the following components are used by the Rust CI to build the target,
and the versions can be seen as the minimum requirement:

* GNU Binutils 2.42
* GCC 14.x
* glibc 2.36
* linux-headers 5.19

Of these, glibc and linux-headers are at their respective earliest versions with
mainline LoongArch support, so it is impossible to use older versions of these.
Older versions of Binutils and GCC will not work either, due to lack of support
for newer LoongArch ELF relocation types, among other features.

Recent LLVM/Clang toolchains may be able to build the targets, but are not
currently being actively tested.

### CPU features

These targets require the double-precision floating-point and LSX (LoongArch
SIMD Extension) features.

## Building

These targets are distributed through `rustup`, and otherwise require no
special configuration.

If you need to build your own Rust for some reason though, the targets can be
simply enabled in `bootstrap.toml`. For example:

```toml
[build]
target = ["loongarch64-unknown-linux-gnu"]
```

Make sure the LoongArch toolchain binaries are reachable from `$PATH`.
Alternatively, you can explicitly configure the paths in `bootstrap.toml`:

```toml
[target.loongarch64-unknown-linux-gnu]
# Adjust the paths to point at your toolchain
# Suppose the toolchain is placed at /TOOLCHAIN_PATH, and the cross prefix is
# "loongarch64-unknown-linux-gnu-":
cc = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-gcc"
cxx = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-g++"
ar = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-ar"
ranlib = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-ranlib"
linker = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-gcc"
```

### Cross-compilation

This target can be cross-compiled on a `x86_64-unknown-linux-gnu` host.
Other hosts are also likely to work, but not actively tested.

You can test the cross build directly on the host, thanks to QEMU linux-user emulation.
An example is given below:

```sh
# Suppose the cross toolchain is placed at $TOOLCHAIN_PATH, with a cross prefix
# of "loongarch64-unknown-linux-gnu-".
export CC_loongarch64_unknown_linux_gnu="$TOOLCHAIN_PATH"/bin/loongarch64-unknown-linux-gnu-gcc
export CXX_loongarch64_unknown_linux_gnu="$TOOLCHAIN_PATH"/bin/loongarch64-unknown-linux-gnu-g++
export AR_loongarch64_unknown_linux_gnu="$TOOLCHAIN_PATH"/bin/loongarch64-unknown-linux-gnu-gcc-ar
export CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PATH"/bin/loongarch64-unknown-linux-gnu-gcc

# Point qemu-loongarch64 to the LoongArch sysroot.
# Suppose the sysroot is located at "sysroot" below the toolchain root:
export CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-loongarch64 -L $TOOLCHAIN_PATH/sysroot"
# Or alternatively, if binfmt_misc is set up for running LoongArch binaries
# transparently:
export QEMU_LD_PREFIX="$TOOLCHAIN_PATH"/sysroot

cargo run --target loongarch64-unknown-linux-gnu --release
```

## Testing

There are no special requirements for testing and running the targets.
For testing cross builds on the host, please refer to the "Cross-compilation"
section above.

## Building Rust programs

As the targets are available through `rustup`, it is very easy to build Rust
programs for these targets: same as with other architectures.
Note that you will need a LoongArch C/C++ toolchain for linking, or if you want
to compile C code along with Rust (such as for Rust crates with C dependencies).

```sh
rustup target add loongarch64-unknown-linux-gnu
cargo build --target loongarch64-unknown-linux-gnu
```

Availability of pre-built artifacts through `rustup` are as follows:

* `loongarch64-unknown-linux-gnu`: since Rust 1.71;
* `loongarch64-unknown-linux-musl`: since Rust 1.81.
