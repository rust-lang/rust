# `riscv64a23-unknown-linux-gnu`

**Tier: 3**

RISC-V target using the ratified [RVA23 Profile](https://github.com/riscv/riscv-profiles/blob/main/src/rva23-profile.adoc).
This target will enable all mandary features of rva23u64 by default.

## Target maintainers

[@ZhongyaoChen](https://github.com/ZhongyaoChen)
[@CaiWeiran](https://github.com/CaiWeiran)

## Requirements

This target can be sucessfully build on the following platform: ubuntu 24.04 (Linux Kernel version 6.8.0, glibc 2.39).

Other platforms may work, but are not tested. Please contanct if you encounter any issues.

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
