# `riscv32{i,im,ima,imc,imfc,imac,imafc}-unknown-none-elf`

**Tier: 2**

Bare-metal target for RISC-V CPUs with the RV32I, RV32IM, RV32IMC, RV32IMAFC and RV32IMAC ISAs.

**Tier: 3**

Bare-metal target for RISC-V CPUs with the RV32IMA and RV32IMFC ISAs.

The `riscv32imfc-unknown-none-elf` target covers RV32IMFC cores that have
hardware single-precision floating point (the `F` extension, using the `ilp32f`
ABI) but *no* atomic (`A`) extension. Like `riscv32imc-unknown-none-elf`, it is
built with `+forced-atomics`: atomic loads/stores lower to plain loads/stores
(sound on a single hart) and `lr`/`sc`/`amo*` instructions are never emitted, so
it does not trap on a core without the `A` extension. Compare-and-swap and other
read-modify-write atomics are disabled (`atomic_cas = false`); downstream crates
that need them use a critical-section polyfill (e.g. `portable-atomic`).

## Target maintainers

* Rust Embedded Working Group, [RISC-V team](https://github.com/rust-embedded/wg#the-risc-v-team)

The `riscv32imfc-unknown-none-elf` target is additionally maintained by:

* [@sanchuanhehe](https://github.com/sanchuanhehe)

## Requirements

The target is cross-compiled, and uses static linking. No external toolchain
is required and the default `rust-lld` linker works, but you must specify
a linker script. The [`riscv-rt`] crate provides a suitable one. The
[`riscv-rust-quickstart`] repository gives an example of an RV32 project.

[`riscv-rt`]: https://crates.io/crates/riscv-rt
[`riscv-rust-quickstart`]: https://github.com/riscv-rust/riscv-rust-quickstart

## Building the target

This target is included in Rust and can be installed via `rustup`.

## Testing

This is a cross-compiled `no-std` target, which must be run either in a simulator
or by programming them onto suitable hardware. It is not possible to run the
Rust test-suite on this target.

## Cross-compilation toolchains and C code

This target supports C code. If interlinking with C or C++, you may need to use
`riscv32-unknown-elf-gcc` as a linker instead of `rust-lld`.
