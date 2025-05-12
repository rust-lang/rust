# riscv32imac-unknown-xous-elf

**Tier: 3**

Xous microkernel, message-based operating system that powers devices such as Precursor and Betrusted. The operating system is written entirely in Rust, so no additional software is required to compile programs for Xous.

## Target maintainers

[@xobs](https://github.com/xobs)

## Requirements


Building the target itself requires a RISC-V compiler that is supported by `cc-rs`. For example, you can use the prebuilt [xPack](https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/latest) toolchain.

Cross-compiling programs does not require any additional software beyond the toolchain. Prebuilt versions of the toolchain are available [from Betrusted](https://github.com/betrusted-io/rust/releases).

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["riscv32imac-unknown-xous-elf"]
```

Make sure your C compiler is included in `$PATH`, then add it to the `bootstrap.toml`:

```toml
[target.riscv32imac-unknown-xous-elf]
cc = "riscv-none-elf-gcc"
ar = "riscv-none-elf-ar"
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will need to do one of the following:

* Build Rust with the target enabled (see "Building the target" above)
* Build your own copy of `core` by using `build-std` or similar
* Download a prebuilt toolchain [from Betrusted](https://github.com/betrusted-io/rust/releases)

## Cross-compilation

This target can be cross-compiled from any host.

## Testing

Currently there is no support to run the rustc test suite for this target.
