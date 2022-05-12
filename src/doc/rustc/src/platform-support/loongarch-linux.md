# loongarch\*-unknown-linux-\*

**Tier: 3**

[LoongArch] is a new RISC ISA developed by Loongson Technology, based in China.

[LoongArch]: https://loongson.github.io/LoongArch-Documentation/README-EN.html

The target name follow this format: `<machine>-<vendor>-<os><fabi_suffix>, where `<machine>` specifies the CPU family/model, `<vendor>` specifies the vendor and `<os>` the operating system name.
While the integer base ABI is implied by the machine field, the floating point base ABI type is encoded into the os field of the specifier using the string suffix `<fabi-suffix>`.

|    `<fabi-suffix>`     |                           `Description`                            |
|------------------------|--------------------------------------------------------------------|
|          f64           | The base ABI use 64-bits FPRs for parameter passing.(lp64d, ilp32d)|
|          f32           | The base ABI uses 32-bit FPRs for parameter passing. (lp64f ilp32f)|
|          sf            | The base ABI uses no FPR for parameter passing. (lp64s ilp32s)     |

|`ABI type(Base ABI/ABI extension)`| `C library` | `kernel` |          `target tuple`          |
|----------------------------------|-------------|----------|----------------------------------|
|           lp64d/base             |   glibc     |  linux   | loongarch64-unknown-linux-gnuf64 |
|           lp64f/base             |   glibc     |  linux   | loongarch64-unknown-linux-gnuf32 |
|           lp64s/base             |   glibc     |  linux   | loongarch64-unknown-linux-gnusf  |
|           ilp32d/base            |   glibc     |  linux   | loongarch32-unknown-linux-gnuf64 |
|           ilp32f/base            |   glibc     |  linux   | loongarch32-unknown-linux-gnuf32 |
|           ilp32s/base            |   glibc     |  linux   | loongarch32-unknown-linux-gnusf  |
|           lp64d/base             |  musl libc  |  linux   | loongarch64-unknown-linux-muslf64|
|           lp64f/base             |  musl libc  |  linux   | loongarch64-unknown-linux-muslf32|
|           lp64s/base             |  musl libc  |  linux   | loongarch64-unknown-linux-muslsf |
|           ilp32d/base            |  musl libc  |  linux   | loongarch32-unknown-linux-muslf64|
|           ilp32f/base            |  musl libc  |  linux   | loongarch32-unknown-linux-muslf32|
|           ilp32s/base            |  musl libc  |  linux   | loongarch32-unknown-linux-muslsf |

## Target maintainers

- zhaixiaojuan, `zhaixiaojuan@loongson.cn`, https://github.com/zhaixiaojuan

## Requirements

This target is cross-compiled.
A GNU toolchain for LoongArch target is required.  It can be downloaded from https://github.com/loongson/build-tools/releases, or built from the source code of GCC (12.1.0 or later) and Binutils (2.39 or later).

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["loongarch64-unknown-linux-gnuf64"]
```

Make sure `loongarch64-unknown-linux-gnu-gcc` can be searched from the directories specified in`$PATH`. Alternatively, you can use GNU LoongArch Toolchain by adding the following to `config.toml`:

```toml
[target.loongarch64-unknown-linux-gnuf64]
# ADJUST THIS PATH TO POINT AT YOUR TOOLCHAIN
cc = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-gcc"
cxx = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-g++"
ar = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-ar"
ranlib = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-ranlib"
linker = "/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-gcc"
```

## Cross-compilation

This target can be cross-compiled on a `x86_64-unknown-linux-gnu` host. Cross-compilation on other hosts may work but is not tested.

## Testing
To test a cross-compiled binary on your build system, install the qemu binary that supports the LoongArch architecture and execute the following commands.
```text
CC_loongarch64_unknown_linux_gnu=/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-gcc \
CXX_loongarch64_unknown_linux_gnu=/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-g++ \
AR_loongarch64_unknown_linux_gnu=/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-gcc-ar \
CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNUN_LINKER=/TOOLCHAIN_PATH/bin/loongarch64-unknown-linux-gnu-gcc \
# SET TARGET SYSTEM LIBYARY PATH
CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNUN_RUNNER="qemu-loongarch64 -L /TOOLCHAIN_PATH/TARGET_LIBRAY_PATH" \
cargo run --target loongarch64-unknown-linux-gnuf64 --release
```
Tested on x86 architecture, other architectures not tested.
The qemu-loongarch64 binary download address that can run on the x86_64: https://github.com/loongson/build-tools


## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for this target, you will either need to build Rust with the target enabled (see "Building the target" above), or build your own copy of `std` by using `build-std` or similar.

If `rustc` has support for that target and the library artifacts are available, then Rust static libraries can be built for that target:

```shell
$ rustc --target loongarch64-unknown-linux-gnuf64 your-code.rs --crate-type staticlib
$ ls libyour_code.a
```

On Rust Nightly it's possible to build without the target artifacts available:

```text
cargo build -Z build-std --target loongarch64-unknown-linux-gnuf64
```
