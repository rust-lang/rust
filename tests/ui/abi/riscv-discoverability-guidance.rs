// ignore-tidy-linelength
//@ revisions: riscv32 riscv64
//
//@ [riscv32] needs-llvm-components: riscv
//@ [riscv32] compile-flags: --target=riscv32i-unknown-none-elf -C target-feature=-unaligned-scalar-mem --crate-type=rlib
//@ [riscv64] needs-llvm-components: riscv
//@ [riscv64] compile-flags: --target=riscv64gc-unknown-none-elf -C target-feature=-unaligned-scalar-mem --crate-type=rlib
#![no_core]
#![feature(
    no_core,
    lang_items,
    abi_riscv_interrupt
)]
#[lang = "sized"]
trait Sized {}

extern "riscv-interrupt" fn isr() {}
//~^ ERROR invalid ABI
//~^^ NOTE invalid ABI
//~^^^ NOTE invoke `rustc --print=calling-conventions` for a full list of supported calling conventions

extern "riscv-interrupt-u" fn isr_U() {}
//~^ ERROR invalid ABI
//~^^ NOTE invalid ABI
//~^^^ NOTE invoke `rustc --print=calling-conventions` for a full list of supported calling conventions
