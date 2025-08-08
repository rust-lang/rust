# `riscv64a23-unknown-linux-gnu`

**Tier: 3**

RISC-V target using the ratified [RVA23 Profile](https://github.com/riscv/riscv-profiles/blob/main/src/rva23-profile.adoc).
This target will enable all mandary features of rva23u64 and rva23s64 by default.

## Target maintainers

[@ZhongyaoChen](https://github.com/ZhongyaoChen)
[@CaiWeiran](https://github.com/CaiWeiran)

## Requirements

This target requires:

* Linux Kernel version 4.20 or later
* glibc 2.17 or later

## Building the target

Tier-3 target is not distributed through `rustup`.

You need to build your own Rust, the target can be build with:

```bash
./x build --target riscv64a23-unknown-linux-gnu
```

## Building Rust programs

Add the toolchain:

```bash
rustup toolchain link rva23-toolchain {path-to-rust}/build/host/stage2
```

Then cross compile crates with:

```bash
RUSTFLAGS="-C linker=riscv64-linux-gnu-gcc" cargo +rva23-toolchain build --target=riscv64a23-unknown-linux-gnu
```
