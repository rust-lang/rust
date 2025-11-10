# `riscv64im-unknown-none-elf`

**Tier: 3**

Bare-metal target for RISC-V CPUs with the RV64IM ISA.

## Target maintainers

* Rust Embedded Working Group, [RISC-V team](https://github.com/rust-embedded/wg#the-risc-v-team)

## Requirements

This target is cross-compiled and uses static linking. The target supports `core` and `alloc`, but not `std`.

The target does not support atomic compare-and-swap operations, as the RV64IM ISA lacks the "A" (Atomics) extension. Atomic operations are emulated using the `+forced-atomics` feature.

No external toolchain is required and the default `rust-lld` linker works, but you must specify a linker script. The [`riscv-rt`] crate provides suitable linker scripts. The [`riscv-rust-quickstart`] repository gives examples of RISC-V bare-metal projects.

[`riscv-rt`]: https://crates.io/crates/riscv-rt
[`riscv-rust-quickstart`]: https://github.com/riscv-rust/riscv-rust-quickstart

## Building the target

This target is included in Rust and can be installed via `rustup`:

```bash
rustup target add riscv64im-unknown-none-elf
```

## Building Rust programs

Build using the standard Cargo workflow:

```bash
cargo build --target riscv64im-unknown-none-elf
```

You will need to provide a linker script. The [`riscv-rt`] crate handles this automatically when used as a dependency.

## Testing

This is a cross-compiled `no-std` target, which must be run either in a simulator or by programming onto suitable hardware. It is not possible to run the Rust test-suite on this target.

You can test the target in QEMU with:

```bash
qemu-system-riscv64 -machine virt -cpu rv64,a=false,c=false -nographic -semihosting -kernel your-binary
```

Note: You must explicitly disable the 'a' (atomics) and 'c' (compressed) extensions when using QEMU to accurately emulate an RV64IM-only CPU.

## Cross-compilation toolchains and C code

This target supports C code. If interlinking with C or C++, you may need to use `riscv64-unknown-elf-gcc` with the appropriate `-march=rv64im -mabi=lp64` flags as a linker instead of `rust-lld`.
