# `riscv64im-unknown-none-elf`

**Tier: 3**

Bare-metal target for RISC-V CPUs with the RV64IM ISA.

## Target maintainers

* Rust Embedded Working Group, [RISC-V team](https://github.com/rust-embedded/wg#the-risc-v-team)

## Requirements

This target is cross-compiled and uses static linking. The target supports `core` and `alloc`, but not `std`.

As the RV64IM ISA lacks the "A" (Atomics) extension, atomic operations are emulated using the `+forced-atomics` feature.

No external toolchain is required and the default `rust-lld` linker works, but you must specify a linker script. The [`riscv-rt`] crate provides suitable linker scripts. The [`riscv-rust-quickstart`] repository gives examples of RISC-V bare-metal projects.

[`riscv-rt`]: https://crates.io/crates/riscv-rt
[`riscv-rust-quickstart`]: https://github.com/riscv-rust/riscv-rust-quickstart

## Building the target

You can build Rust with support for the target by adding it to the `target` list in `bootstrap.toml`:

```toml
[build]
target = ["riscv64im-unknown-none-elf"]
```

Alternatively, you can use the `-Z build-std` flag to build the standard library on-demand:

```bash
cargo build -Z build-std=core,alloc --target riscv64im-unknown-none-elf
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for this target (see "Building the target" above)

## Testing

This is a cross-compiled `no-std` target, which must be run either in a simulator or by programming onto suitable hardware. It is not possible to run the Rust test-suite on this target.

## Cross-compilation toolchains and C code

This target supports C code. If interlinking with C or C++, you may need to use `riscv64-unknown-elf-gcc` with the appropriate `-march=rv64im -mabi=lp64` flags as a linker instead of `rust-lld`.
