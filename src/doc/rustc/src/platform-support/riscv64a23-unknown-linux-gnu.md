# `riscv64a23-unknown-linux-gnu`

**Tier: 3**

RISC-V targets using the ratified [RVA23 Profile](https://github.com/riscv/riscv-profiles/blob/main/rva23-profile.adoc).
This target will enable all mandary features of rva23u64 and rva23s64 by default.


## Target maintainers

[@ZhongyaoChen](https://github.com/ZhongyaoChen)
[@CaiWeiran](https://github.com/CaiWeiran)

## Requirements

This target requires:

* Linux Kernel version 4.20 or later
* glibc 2.17 or later


## Building the target

The target is distributed through `rustup`, and otherwise require no
special configuration.

If you need to build your own Rust for some reason though, the target can be build with:

```bash
./x build --target riscv64a23-unknown-linux-gnu
```

## Building Rust programs

Add the target:

```bash
rustup target add riscv64a23-unknown-linux-gnu
```

Then cross compile crates with:

```bash
cargo build --target riscv64a23-unknown-linux-gnu
```

## Cross-compilation toolchains and Testing

On Ubuntu 24.04, we can install compilation dependencies with:

```bash
apt install -y git python3 g++ g++-riscv64-linux-gnu
```

Then build target with:

```bash
./x build --target=riscv64a23-unknown-linux-gnu
```

There are no special requirements for testing and running the targets.
For testing cross-builds on the host, you can use the docker image. It will automatically set up a RISC-V QEMU emulator and run all the test suite.

```bash
DEPLOY=1 ./src/ci/docker/run.sh riscv64a23-gnu
```
