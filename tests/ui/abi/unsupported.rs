//@ add-core-stubs
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
    abi_gpu_kernel,
    abi_x86_interrupt,
    abi_riscv_interrupt,
    abi_c_cmse_nonsecure_call,
    cmse_nonsecure_entry
)]

extern crate minicore;
use minicore::*;

extern "ptx-kernel" fn ptx() {}
//~^ ERROR is not a supported ABI
fn ptx_ptr(f: extern "ptx-kernel" fn()) {
//~^ ERROR is not a supported ABI
    f()
}
extern "ptx-kernel" {}
//~^ ERROR is not a supported ABI
extern "gpu-kernel" fn gpu() {}
//~^ ERROR is not a supported ABI

extern "aapcs" fn aapcs() {}
//[x64]~^ ERROR is not a supported ABI
//[i686]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI
fn aapcs_ptr(f: extern "aapcs" fn()) {
//[x64]~^ ERROR is not a supported ABI
//[i686]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI
    f()
}
extern "aapcs" {}
//[x64]~^ ERROR is not a supported ABI
//[i686]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI

extern "msp430-interrupt" fn msp430() {}
//~^ ERROR is not a supported ABI
fn msp430_ptr(f: extern "msp430-interrupt" fn()) {
//~^ ERROR is not a supported ABI
    f()
}
extern "msp430-interrupt" {}
//~^ ERROR is not a supported ABI

extern "avr-interrupt" fn avr() {}
//~^ ERROR is not a supported ABI
fn avr_ptr(f: extern "avr-interrupt" fn()) {
//~^ ERROR is not a supported ABI
    f()
}
extern "avr-interrupt" {}
//~^ ERROR is not a supported ABI

extern "riscv-interrupt-m" fn riscv() {}
//[arm]~^ ERROR is not a supported ABI
//[x64]~^^ ERROR is not a supported ABI
//[i686]~^^^ ERROR is not a supported ABI
//[aarch64]~^^^^ ERROR is not a supported ABI
fn riscv_ptr(f: extern "riscv-interrupt-m" fn()) {
//[arm]~^ ERROR is not a supported ABI
//[x64]~^^ ERROR is not a supported ABI
//[i686]~^^^ ERROR is not a supported ABI
//[aarch64]~^^^^ ERROR is not a supported ABI
    f()
}
extern "riscv-interrupt-m" {}
//[arm]~^ ERROR is not a supported ABI
//[x64]~^^ ERROR is not a supported ABI
//[i686]~^^^ ERROR is not a supported ABI
//[aarch64]~^^^^ ERROR is not a supported ABI

extern "x86-interrupt" fn x86() {}
//[aarch64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
//[riscv32]~^^^ ERROR is not a supported ABI
//[riscv64]~^^^^ ERROR is not a supported ABI
fn x86_ptr(f: extern "x86-interrupt" fn()) {
//[aarch64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
//[riscv32]~^^^ ERROR is not a supported ABI
//[riscv64]~^^^^ ERROR is not a supported ABI
    f()
}
extern "x86-interrupt" {}
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
fn thiscall_ptr(f: extern "thiscall" fn()) {
//[x64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI
    f()
}
extern "thiscall" {}
//[x64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI

extern "stdcall" fn stdcall() {}
//[x64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI
fn stdcall_ptr(f: extern "stdcall" fn()) {
//[x64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI
    f()
}
extern "stdcall" {}
//[x64]~^ ERROR is not a supported ABI
//[arm]~^^ ERROR is not a supported ABI
//[aarch64]~^^^ ERROR is not a supported ABI
//[riscv32]~^^^^ ERROR is not a supported ABI
//[riscv64]~^^^^^ ERROR is not a supported ABI

fn cmse_call_ptr(f: extern "C-cmse-nonsecure-call" fn()) {
//~^ ERROR is not a supported ABI
    f()
}

extern "C-cmse-nonsecure-entry" fn cmse_entry() {}
//~^ ERROR is not a supported ABI
fn cmse_entry_ptr(f: extern "C-cmse-nonsecure-entry" fn()) {
//~^ ERROR is not a supported ABI
    f()
}
extern "C-cmse-nonsecure-entry" {}
//~^ ERROR is not a supported ABI
