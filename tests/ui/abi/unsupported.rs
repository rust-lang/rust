//@ revisions: x64 i686 aarch64 arm riscv32 riscv64
//
//@ [x64] needs-llvm-components: x86
//@ [x64] compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
//@ [i686] needs-llvm-components: x86
//@ [i686] compile-flags: --target=i686-unknown-linux-gnu --crate-type=rlib
//@ [aarch64] needs-llvm-components: aarch64
//@ [aarch64] compile-flags: --target=aarch64-unknown-linux-gnu --crate-type=rlib
//@ [arm] needs-llvm-components: arm
//@ [arm] compile-flags: --target=armv7-unknown-linux-gnueabihf --crate-type=rlib
//@ [riscv32] needs-llvm-components: riscv
//@ [riscv32] compile-flags: --target=riscv32i-unknown-none-elf --crate-type=rlib
//@ [riscv64] needs-llvm-components: riscv
//@ [riscv64] compile-flags: --target=riscv64gc-unknown-none-elf --crate-type=rlib
#![no_core]
#![feature(
    no_core,
    lang_items,
    abi_ptx,
    abi_msp430_interrupt,
    abi_avr_interrupt,
    wasm_abi,
    abi_x86_interrupt,
    abi_riscv_interrupt
)]
#[lang = "sized"]
trait Sized {}

extern "ptx-kernel" fn ptx() {}
//~^ ERROR is not a supported ABI
extern "wasm" fn wasm() {}
//~^ ERROR is not a supported ABI
extern "aapcs" fn aapcs() {}
//[x64]~^ ERROR is not a supported ABI
//[i686]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI
extern "msp430-interrupt" fn msp430() {}
//~^ ERROR is not a supported ABI
extern "avr-interrupt" fn avr() {}
//~^ ERROR is not a supported ABI
extern "riscv-interrupt-m" fn riscv() {}
//[arm]~^ ERROR is not a supported ABI
//[x64]~^^ ERROR is not a supported ABI
//[i686]~^^^ ERROR is not a supported ABI
//[aarch64]~^^^^ ERROR is not a supported ABI
extern "x86-interrupt" fn x86() {}
//[aarch64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
//[riscv32]~^^^ ERROR is not a supported ABI
//[riscv64]~^^^^ ERROR is not a supported ABI
extern "thiscall" fn thiscall() {}
//[x64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI
extern "stdcall" fn stdcall() {}
//[x64]~^ WARN use of calling convention not supported
//[x64]~^^ WARN this was previously accepted
//[arm]~^^^ WARN use of calling convention not supported
//[arm]~^^^^ WARN this was previously accepted
//[aarch64]~^^^^^ WARN use of calling convention not supported
//[aarch64]~^^^^^^ WARN this was previously accepted
//[riscv32]~^^^^^^^ WARN use of calling convention not supported
//[riscv32]~^^^^^^^^ WARN this was previously accepted
//[riscv64]~^^^^^^^^^ WARN use of calling convention not supported
//[riscv64]~^^^^^^^^^^ WARN this was previously accepted
