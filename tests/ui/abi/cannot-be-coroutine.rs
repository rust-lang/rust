//@ add-core-stubs
//@ edition: 2021
//@ revisions: x64 x64_win i686 riscv32 riscv64 avr msp430
//
//@ [x64] needs-llvm-components: x86
//@ [x64] compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
//@ [x64_win] needs-llvm-components: x86
//@ [x64_win] compile-flags: --target=x86_64-pc-windows-msvc --crate-type=rlib
//@ [i686] needs-llvm-components: x86
//@ [i686] compile-flags: --target=i686-unknown-linux-gnu --crate-type=rlib
//@ [riscv32] needs-llvm-components: riscv
//@ [riscv32] compile-flags: --target=riscv32i-unknown-none-elf --crate-type=rlib
//@ [riscv64] needs-llvm-components: riscv
//@ [riscv64] compile-flags: --target=riscv64gc-unknown-none-elf --crate-type=rlib
//@ [avr] needs-llvm-components: avr
//@ [avr] compile-flags: --target=avr-none -C target-cpu=atmega328p --crate-type=rlib
//@ [msp430] needs-llvm-components: msp430
//@ [msp430] compile-flags: --target=msp430-none-elf --crate-type=rlib
#![no_core]
#![feature(
    no_core,
    abi_msp430_interrupt,
    abi_avr_interrupt,
    abi_x86_interrupt,
    abi_riscv_interrupt
)]

extern crate minicore;
use minicore::*;

// We ignore this error; implementing all of the async-related lang items is not worth it.
async fn vanilla(){
    //~^ ERROR requires `ResumeTy` lang_item
}

async extern "avr-interrupt" fn avr() {
    //[avr]~^ ERROR functions with the "avr-interrupt" ABI cannot be `async`
}

async extern "msp430-interrupt" fn msp430() {
    //[msp430]~^ ERROR functions with the "msp430-interrupt" ABI cannot be `async`
}

async extern "riscv-interrupt-m" fn riscv_m() {
    //[riscv32,riscv64]~^ ERROR functions with the "riscv-interrupt-m" ABI cannot be `async`
}

async extern "riscv-interrupt-s" fn riscv_s() {
    //[riscv32,riscv64]~^ ERROR functions with the "riscv-interrupt-s" ABI cannot be `async`
}

async extern "x86-interrupt" fn x86(_p: *mut ()) {
    //[x64,x64_win,i686]~^ ERROR functions with the "x86-interrupt" ABI cannot be `async`
}
