/*! Tests entry-point ABIs cannot be called

Interrupt ABIs share similar semantics, in that they are special entry-points unusable by Rust.
So we test that they error in essentially all of the same places.
*/
//@ add-core-stubs
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
    abi_gpu_kernel,
    abi_x86_interrupt,
    abi_riscv_interrupt,
)]

extern crate minicore;
use minicore::*;

/* extern "interrupt" definition */

extern "msp430-interrupt" fn msp430() {}
//[x64,x64_win,i686,riscv32,riscv64,avr]~^ ERROR is not a supported ABI
extern "avr-interrupt" fn avr() {}
//[x64,x64_win,i686,riscv32,riscv64,msp430]~^ ERROR is not a supported ABI
extern "riscv-interrupt-m" fn riscv_m() {}
//[x64,x64_win,i686,avr,msp430]~^ ERROR is not a supported ABI
extern "riscv-interrupt-s" fn riscv_s() {}
//[x64,x64_win,i686,avr,msp430]~^ ERROR is not a supported ABI
extern "x86-interrupt" fn x86() {}
//[riscv32,riscv64,avr,msp430]~^ ERROR is not a supported ABI

/* extern "interrupt" calls  */
fn call_the_interrupts() {
    msp430();
    //[msp430]~^ ERROR functions with the "msp430-interrupt" ABI cannot be called
    avr();
    //[avr]~^ ERROR functions with the "avr-interrupt" ABI cannot be called
    riscv_m();
    //[riscv32,riscv64]~^ ERROR functions with the "riscv-interrupt-m" ABI cannot be called
    riscv_s();
    //[riscv32,riscv64]~^ ERROR functions with the "riscv-interrupt-s" ABI cannot be called
    x86();
    //[x64,x64_win,i686]~^ ERROR functions with the "x86-interrupt" ABI cannot be called
}

/* extern "interrupt" fnptr calls */

fn msp430_ptr(f: extern "msp430-interrupt" fn()) {
    //[x64,x64_win,i686,riscv32,riscv64,avr]~^ ERROR is not a supported ABI
    f()
    //[msp430]~^ ERROR functions with the "msp430-interrupt" ABI cannot be called
}

fn avr_ptr(f: extern "avr-interrupt" fn()) {
    //[x64,x64_win,i686,riscv32,riscv64,msp430]~^ ERROR is not a supported ABI
    f()
    //[avr]~^ ERROR functions with the "avr-interrupt" ABI cannot be called
}

fn riscv_m_ptr(f: extern "riscv-interrupt-m" fn()) {
    //[x64,x64_win,i686,avr,msp430]~^ ERROR is not a supported ABI
    f()
    //[riscv32,riscv64]~^ ERROR functions with the "riscv-interrupt-m" ABI cannot be called
}

fn riscv_s_ptr(f: extern "riscv-interrupt-s" fn()) {
    //[x64,x64_win,i686,avr,msp430]~^ ERROR is not a supported ABI
    f()
    //[riscv32,riscv64]~^ ERROR functions with the "riscv-interrupt-s" ABI cannot be called
}

fn x86_ptr(f: extern "x86-interrupt" fn()) {
    //[riscv32,riscv64,avr,msp430]~^ ERROR is not a supported ABI
    f()
    //[x64,x64_win,i686]~^ ERROR functions with the "x86-interrupt" ABI cannot be called
}
