// ignore-tidy-linelength
//@ add-minicore
//@ revisions: riscv32 riscv64
//
//@ [riscv32] needs-llvm-components: riscv
//@ [riscv32] compile-flags: --target=riscv32i-unknown-none-elf -C target-feature=-unaligned-scalar-mem --crate-type=rlib
//@ [riscv64] needs-llvm-components: riscv
//@ [riscv64] compile-flags: --target=riscv64gc-unknown-none-elf -C target-feature=-unaligned-scalar-mem --crate-type=rlib
//@ ignore-backends: gcc
#![no_core]
#![feature(no_core, lang_items, abi_riscv_interrupt)]

//~? WARN unstable feature specified for `-Ctarget-feature`
//~? NOTE this feature is not stably supported; its behavior can change in the future

extern crate minicore;
use minicore::*;

extern "riscv-interrupt" fn isr() {}
//~^ ERROR invalid ABI
//~^^ NOTE invalid ABI
//~^^^ NOTE invoke `rustc --print=calling-conventions` for a full list of supported calling conventions

extern "riscv-interrupt-u" fn isr_U() {}
//~^ ERROR invalid ABI
//~^^ NOTE invalid ABI
//~^^^ NOTE invoke `rustc --print=calling-conventions` for a full list of supported calling conventions
