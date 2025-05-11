# `csky-unknown-linux-gnuabiv2`

**Tier: 3**

This target supports [C-SKY](https://github.com/c-sky) CPUs with `abi` v2 and `glibc`.

target | std | host | notes
-------|:---:|:----:|-------
`csky-unknown-linux-gnuabiv2` | ✓ |  | C-SKY abiv2 Linux (little endian)
`csky-unknown-linux-gnuabiv2hf` | ✓ |  | C-SKY abiv2 Linux, hardfloat (little endian)

Reference:

- [CSKY ABI Manual](https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1695027452256/T-HEAD_800_Series_ABI_Standards_Manual.pdf)
- [csky-linux-gnuabiv2-toolchain](https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource/1356021/1619528643136/csky-linux-gnuabiv2-tools-x86_64-glibc-linux-4.9.56-20210423.tar.gz)
- [csky-linux-gnuabiv2-qemu](https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1689324918932/xuantie-qemu-x86_64-Ubuntu-18.04-20230714-0202.tar.gz)

other links:

- https://c-sky.github.io/
- https://gitlab.com/c-sky/

## Target maintainers

[@Dirreke](https://github.com/Dirreke)

## Requirements

## Building the target

### Get a C toolchain

Compiling rust for this target has been tested on `x86_64` linux hosts.  Other host types have not been tested, but may work, if you can find a suitable cross compilation toolchain for them.

If you don't already have a suitable toolchain, you can download from [here](https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource/1356021/1619528643136/csky-linux-gnuabiv2-tools-x86_64-glibc-linux-4.9.56-20210423.tar.gz), and unpack it into a directory.

### Configure rust

The target can be built by enabling it for a `rustc` build, by placing the following in `bootstrap.toml`:

```toml
[build]
target = ["x86_64-unknown-linux-gnu", "csky-unknown-linux-gnuabiv2", "csky-unknown-linux-gnuabiv2hf"]
stage = 2

[target.csky-unknown-linux-gnuabiv2]
# ADJUST THIS PATH TO POINT AT YOUR TOOLCHAIN
cc = "${TOOLCHAIN_PATH}/bin/csky-linux-gnuabiv2-gcc"

[target.csky-unknown-linux-gnuabiv2hf]
# ADJUST THIS PATH TO POINT AT YOUR TOOLCHAIN
cc = "${TOOLCHAIN_PATH}/bin/csky-linux-gnuabiv2-gcc"
```

### Build

```sh
# in rust dir
./x.py build --stage 2
```

## Building and Running Rust programs

To test cross-compiled binaries on a `x86_64` system, you can use the `qemu-cskyv2`. This avoids having a full emulated ARM system by doing dynamic binary translation and dynamic system call translation.  It lets you run CSKY programs directly on your `x86_64` kernel.  It's very convenient!

To use:

* Install `qemu-cskyv2` (If you don't already have a qemu, you can download from [here](https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1689324918932/xuantie-qemu-x86_64-Ubuntu-18.04-20230714-0202.tar.gz"), and unpack it into a directory.)
* Link your built toolchain via:
  * `rustup toolchain link stage2 ${RUST}/build/x86_64-unknown-linux-gnu/stage2`
* Create a test program

```sh
cargo new hello_world
cd hello_world
```

* Build and run

```sh
CARGO_TARGET_CSKY_UNKNOWN_LINUX_GNUABIV2_RUNNER=${QEMU_PATH}/bin/qemu-cskyv2 -L ${TOOLCHAIN_PATH}/csky-linux-gnuabiv2/libc \
CARGO_TARGET_CSKY_UNKNOWN_LINUX_GNUABIV2_LINKER=${TOOLCHAIN_PATH}/bin/csky-linux-gnuabiv2-gcc \
RUSTFLAGS="-C target-feature=+crt-static" \
cargo +stage2 run --target csky-unknown-linux-gnuabiv2
```

Attention: The dynamic-linked program may nor be run by `qemu-cskyv2` but can be run on the target.
