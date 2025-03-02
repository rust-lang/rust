# `riscv64gc-unknown-linux-gnu`

**Tier: 2 (with Host Tools)**

RISC-V targets using the *RV64I* base instruction set with the *G* collection of extensions, as well as the *C* extension.


## Target maintainers

- Kito Cheng, <kito.cheng@gmail.com>, [@kito-cheng](https://github.com/kito-cheng)
- Michael Maitland, <michaeltmaitland@gmail.com>, [@michaelmaitland](https://github.com/michaelmaitland)
- Robin Randhawa, <robin.randhawa@sifive.com>, [@robin-randhawa-sifive](https://github.com/robin-randhawa-sifive)
- Craig Topper, <craig.topper@sifive.com>, [@topperc](https://github.com/topperc)

## Requirements

This target requires:

* Linux Kernel version 4.20 or later
* glibc 2.17 or later


## Building the target

These targets are distributed through `rustup`, and otherwise require no
special configuration.

If you need to build your own Rust for some reason though, the targets can be
enabled in `bootstrap.toml`. For example:

```toml
[build]
target = ["riscv64gc-unknown-linux-gnu"]
```


## Building Rust programs


On a RISC-V host, the `riscv64gc-unknown-linux-gnu` target should be automatically
installed and used by default.

On a non-RISC-V host, add the target:

```bash
rustup target add riscv64gc-unknown-linux-gnu
```

Then cross compile crates with:

```bash
cargo build --target riscv64gc-unknown-linux-gnu
```


## Testing

There are no special requirements for testing and running the targets.
For testing cross builds on the host, please refer to the "Cross-compilation
toolchains and C code"
section below.


## Cross-compilation toolchains and C code

A RISC-V toolchain can be obtained for Windows/Mac/Linux from the
[`riscv-gnu-toolchain`](https://github.com/riscv-collab/riscv-gnu-toolchain)
repostory. Binaries are available via
[embecosm](https://www.embecosm.com/resources/tool-chain-downloads/#riscv-linux),
and may also be available from your OS's package manager.

On Ubuntu, a RISC-V toolchain can be installed with:

```bash
apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu libc6-dev-riscv64-cross
```

Depending on your system, you may need to configure the target to use the GNU
GCC linker. To use it, add the following to your `.cargo/config.toml`:

```toml
[target.riscv64gc-unknown-linux-gnu]
linker = "riscv64-linux-gnu-gcc"
```

If your `riscv64-linux-gnu-*` toolchain is not in your `PATH` you may need to
configure additional settings:

```toml
[target.riscv64gc-unknown-linux-gnu]
# Adjust the paths to point at your toolchain
cc = "/TOOLCHAIN_PATH/bin/riscv64-linux-gnu-gcc"
cxx = "/TOOLCHAIN_PATH/bin/riscv64-linux-gnu-g++"
ar = "/TOOLCHAIN_PATH/bin/riscv64-linux-gnu-ar"
ranlib = "/TOOLCHAIN_PATH/bin/riscv64-linux-gnu-ranlib"
linker = "/TOOLCHAIN_PATH/bin/riscv64-linux-gnu-gcc"
```

To test cross compiled binaries on a non-RISCV-V host, you can use
[`qemu`](https://www.qemu.org/docs/master/system/target-riscv.html).
On Ubuntu, a RISC-V emulator can be obtained with:

```bash
apt install qemu-system-riscv64
```

Then, in `.cargo/config.toml` set the `runner`:

```toml
[target.riscv64gc-unknown-linux-gnu]
runner = "qemu-riscv64-static -L /usr/riscv64-linux-gnu -cpu rv64"
```

On Mac and Linux, it's also possible to use
[`lima`](https://github.com/lima-vm/lima) to emulate RISC-V in a similar way to
how WSL2 works on Windows:

```bash
limactl start template://riscv
limactl shell riscv
```

Using [Docker (with BuildKit)](https://docs.docker.com/build/buildkit/) the
[`riscv64/ubuntu`](https://hub.docker.com/r/riscv64/ubuntu) image can be used
to build or run `riscv64gc-unknown-linux-gnu` binaries.

```bash
docker run --platform linux/riscv64 -ti --rm --mount "type=bind,src=$(pwd),dst=/checkout" riscv64/ubuntu bash
```
