# mipsisa\*r6\*-unknown-linux-gnu\*

**Tier: 3**

[MIPS Release 6](https://s3-eu-west-1.amazonaws.com/downloads-mips/documents/MD00083-2B-MIPS64INT-AFP-06.01.pdf), or simply MIPS R6, is the latest iteration of the MIPS instruction set architecture (ISA).

MIPS R6 is experimental in nature, as there is not yet real hardware. However, Qemu emulation is available and we have two Linux distros maintained for development and evaluation purposes. This documentation describes the Rust support for MIPS R6 targets under `mipsisa*r6*-unknown-linux-gnu*`.

The target name follow this format: `<machine>-<vendor>-<os><abi_suffix>`, where `<machine>` specifies the CPU family/model, `<vendor>` specifies the vendor and `<os>` the operating system name. The `<abi_suffix>` denotes the base ABI (32/n32/64/o64).

| ABI suffix | Description                        |
|------------|------------------------------------|
| abi64      | Uses the 64-bit (64) ABI           |
| abin32     | Uses the n32 ABI                   |
| N/A        | Uses the (assumed) 32-bit (32) ABI |

## Target Maintainers

- [Xuan Chen](https://github.com/chenx97) <henry.chen@oss.cipunited.com>
- [Walter Ji](https://github.com/709924470) <walter.ji@oss.cipunited.com>
- [Xinhui Yang](https://github.com/Cyanoxygen) <cyan@oss.cipunited.com>
- [Lain Yang](https://github.com/Fearyncess) <lain.yang@oss.cipunited.com>

## Requirements

### C/C++ Toolchain

A GNU toolchain for one of the MIPS R6 target is required. [AOSC OS](https://aosc.io/) provides working native and cross-compiling build environments. You may also supply your own a toolchain consisting of recent versions of GCC and Binutils.

### Target libraries

A minimum set of libraries is required to perform dynamic linking:

- GNU glibc
- OpenSSL
- Zlib
- Linux API Headers

This set of libraries should be installed to make up minimal target sysroot.

For AOSC OS, You may install such a sysroot with the following commands:

```sh
cd /tmp

# linux+api, glibc, and file system structure are included in the toolchain.
sudo apt install gcc+cross-mips64r6el binutils+cross-mips64r6el

# Download and extract required libraries.
wget https://repo.aosc.io/debs/pool/stable/main/z/zlib_1.2.13-0_mips64r6el.deb -O zlib.deb
wget https://repo.aosc.io/debs/pool/stable/main/o/openssl_1.1.1q-1_mips64r6el.deb -O openssl.deb

# Extract them to your desired location.
for i in zlib openssl ; do
    sudo dpkg-deb -vx $i.deb /var/ab/cross-root/mips64r6el
done

# Workaround a possible ld bug when using -Wl,-Bdynamic.
sudo sed -i 's|/usr|=/usr|g' /var/ab/cross-root/mips64r6el/usr/lib/libc.so
```

For other distros, you may build them manually.

## Building

The following procedure outlines the build process for the MIPS64 R6 target with 64-bit (64) ABI (`mipsisa64r6el-unknown-linux-gnuabi64`).

### Prerequisite: Disable debuginfo

A LLVM bug makes rustc crash if debug or debug info generation is enabled. You need to edit `config.toml` to disable this:

```toml
[rust]
debug = false
debug-info-level = 0
```

### Prerequisite: Enable rustix's libc backend

The crate `rustix` may try to link itself against MIPS R2 assembly, resulting in linkage error. To avoid this, you may force `rustix` to use its fallback `libc` backend by setting relevant `RUSTFLAGS`:

```sh
export RUSTFLAGS="--cfg rustix_use_libc"
```

This will trigger warnings during build, as `-D warnings` is enabled by default. Disable `-D warnings` by editing `config.toml` to append the following:

```toml
[rust]
deny-warnings = false
```

### Prerequisite: Supplying OpenSSL

As a Tier 3 target, `openssl_sys` lacks the vendored OpenSSL library for this target. You will need to provide a prebuilt OpenSSL library to link `cargo`. Since we have a pre-configured sysroot, we can point to it directly:

```sh
export MIPSISA64R6EL_UNKNOWN_LINUX_GNUABI64_OPENSSL_NO_VENDOR=y
export MIPSISA64R6EL_UNKNOWN_LINUX_GNUABI64_OPENSSL_DIR="/var/ab/cross-root/mips64r6el/usr"
```

On Debian, you may need to provide library path and include path separately:

```sh
export MIPSISA64R6EL_UNKNOWN_LINUX_GNUABI64_OPENSSL_NO_VENDOR=y
export MIPSISA64R6EL_UNKNOWN_LINUX_GNUABI64_OPENSSL_LIB_DIR="/usr/lib/mipsisa64r6el-linux-gnuabi64/"
export MIPSISA64R6EL_UNKNOWN_LINUX_GNUABI64_OPENSSL_INCLUDE_DIR="/usr/include"
```

### Launching `x.py`

```toml
[build]
target = ["mipsisa64r6el-unknown-linux-gnuabi64"]
```

Make sure that `mipsisa64r6el-unknown-linux-gnuabi64-gcc` is available from your executable search path (`$PATH`).

Alternatively, you can specify the directories to all necessary toolchain executables in `config.toml`:

```toml
[target.mipsisa64r6el-unknown-linux-gnuabi64]
# Adjust the paths below to point to your toolchain installation prefix.
cc = "/toolchain_prefix/bin/mipsisa64r6el-unknown-linux-gnuabi64-gcc"
cxx = "/toolchain_prefix/bin/mipsisa64r6el-unknown-linux-gnuabi64-g++"
ar = "/toolchain_prefix/bin/mipsisa64r6el-unknown-linux-gnuabi64-gcc-ar"
ranlib = "/toolchain_prefix/bin/mipsisa64r6el-unknown-linux-gnuabi64-ranlib"
linker = "/toolchain_prefix/bin/mipsisa64r6el-unknown-linux-gnuabi64-gcc"
```

Or, you can specify your cross compiler toolchain with an environment variable:

```sh
export CROSS_COMPILE="/opt/abcross/mips64r6el/bin/mipsisa64r6el-aosc-linux-gnuabi64-"
```

Finally, launch the build script:

```sh
./x.py build
```

### Tips

- Avoid setting `cargo-native-static` to `false`, as this will result in a redundant artifact error while building clippy:
    ```text
    duplicate artifacts found when compiling a tool, this typically means that something was recompiled because a transitive dependency has different features activated than in a previous build:

    the following dependencies have different features:
        syn 2.0.8 (registry+https://github.com/rust-lang/crates.io-index)
    `clippy-driver` additionally enabled features {"full"} at ...
    `cargo` additionally enabled features {} at ...

    to fix this you will probably want to edit the local src/tools/rustc-workspace-hack/Cargo.toml crate, as that will update the dependency graph to ensure that these crates all share the same feature set
    thread 'main' panicked at 'tools should not compile multiple copies of the same crate', tool.rs:250:13
    note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
    ```

## Building Rust programs

To build Rust programs for MIPS R6 targets, for instance, the `mipsisa64r6el-unknown-linux-gnuabi64` target:

```bash
cargo build --target mipsisa64r6el-unknown-linux-gnuabi64
```

## Testing

To test a cross-compiled binary on your build system, install the Qemu user emulator that support the MIPS R6 architecture (`qemu-user-mipsel` or `qemu-user-mips64el`). GCC runtime libraries (`libgcc_s`) for the target architecture should be present in target sysroot to run the program.

```sh
env \
    CARGO_TARGET_MIPSISA64R6EL_UNKNOWN_LINUX_GNUABI64_LINKER="/opt/abcross/mips64r6el/bin/mipsisa64r6el-aosc-linux-gnuabi64-gcc" \
    CARGO_TARGET_MIPSISA64R6EL_UNKNOWN_LINUX_GNUABI64_RUNNER="qemu-mips64el-static -L /var/ab/cross-root/mips64r6el" \
    cargo run --release \
        --target mipsisa64r6el-unknown-linux-gnuabi64
```

## Tips for building Rust programs for MIPS R6

- Until we finalize a fix, please make sure the aforementioned workarounds for `rustix` crate and LLVM are always applied. This can be achieved by setting the relevant environment variables, and editing `Cargo.toml` before building.
