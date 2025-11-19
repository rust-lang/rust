# `riscv64a23-unknown-linux-gnu`

**Tier: 2 (without Host Tools)**

RISC-V target using the ratified [RVA23 Profile](https://github.com/riscv/riscv-profiles/blob/main/src/rva23-profile.adoc).
This target will enable all mandary features of rva23u64 by default.

## Target maintainers

[@ZhongyaoChen](https://github.com/ZhongyaoChen)
[@CaiWeiran](https://github.com/CaiWeiran)

## Requirements

This target can be sucessfully build on the following platform: ubuntu 24.04 (Linux Kernel version 6.8.0, glibc 2.39).

Other platforms may work, but are not tested. Please contanct if you encounter any issues.

## Building the target

Tier-2 targets are distributed through `rustup`. Install the target with:

```bash
rustup target add riscv64a23-unknown-linux-gnu
```

## Building Rust programs

Cross compile crates with:

```bash
cargo build --target=riscv64a23-unknown-linux-gnu
```

For cross-compilation, you may need to install the appropriate linker:

```bash
# Ubuntu/Debian
sudo apt-get install gcc-riscv64-linux-gnu

# Then set the linker
RUSTFLAGS="-C linker=riscv64-linux-gnu-gcc" cargo build --target=riscv64a23-unknown-linux-gnu
```
