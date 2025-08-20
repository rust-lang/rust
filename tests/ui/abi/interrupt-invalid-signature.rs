/*! Tests interrupt ABIs have a constricted signature

Most interrupt ABIs share a similar restriction in terms of not allowing most signatures.
Specifically, they generally cannot have arguments or return types.
So we test that they error in essentially all of the same places.
A notable and interesting exception is x86.

This test uses `cfg` because it is not testing whether these ABIs work on the platform.
*/
//@ add-core-stubs
//@ revisions: x64 i686 riscv32 riscv64 avr msp430
//
//@ [x64] needs-llvm-components: x86
//@ [x64] compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
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

/* most extern "interrupt" definitions should not accept args */

#[cfg(msp430)]
extern "msp430-interrupt" fn msp430_arg(_byte: u8) {}
//[msp430]~^ ERROR invalid signature

#[cfg(avr)]
extern "avr-interrupt" fn avr_arg(_byte: u8) {}
//[avr]~^ ERROR invalid signature

#[cfg(any(riscv32,riscv64))]
extern "riscv-interrupt-m" fn riscv_m_arg(_byte: u8) {}
//[riscv32,riscv64]~^ ERROR invalid signature

#[cfg(any(riscv32,riscv64))]
extern "riscv-interrupt-s" fn riscv_s_arg(_byte: u8) {}
//[riscv32,riscv64]~^ ERROR invalid signature


/* all extern "interrupt" definitions should not return non-1ZST values */

#[cfg(avr)]
extern "avr-interrupt" fn avr_ret() -> u8 {
    //[avr]~^ ERROR invalid signature
    1
}

#[cfg(msp430)]
extern "msp430-interrupt" fn msp430_ret() -> u8 {
    //[msp430]~^ ERROR invalid signature
    1
}

#[cfg(any(riscv32,riscv64))]
extern "riscv-interrupt-m" fn riscv_m_ret() -> u8 {
    //[riscv32,riscv64]~^ ERROR invalid signature
    1
}

#[cfg(any(riscv32,riscv64))]
extern "riscv-interrupt-s" fn riscv_s_ret() -> u8 {
    //[riscv32,riscv64]~^ ERROR invalid signature
    1
}

#[cfg(any(x64,i686))]
extern "x86-interrupt" fn x86_ret(_p: *const u8) -> u8 {
    //[x64,i686]~^ ERROR invalid signature
    1
}

#[cfg(any(x64,i686))]
extern "x86-interrupt" fn x86_0() {
    //[x64,i686]~^ ERROR invalid signature
}

#[cfg(any(x64,i686))]
extern "x86-interrupt" fn x86_1(_p1: *const u8) { }

#[cfg(any(x64,i686))]
extern "x86-interrupt" fn x86_2(_p1: *const u8, _p2: *const u8) { }

#[cfg(any(x64,i686))]
extern "x86-interrupt" fn x86_3(_p1: *const u8, _p2: *const u8, _p3: *const u8) {
    //[x64,i686]~^ ERROR invalid signature
}



/* extern "interrupt" fnptrs with invalid signatures */

#[cfg(avr)]
fn avr_ptr(_f: extern "avr-interrupt" fn(u8) -> u8) {
}

#[cfg(msp430)]
fn msp430_ptr(_f: extern "msp430-interrupt" fn(u8) -> u8) {
}

#[cfg(any(riscv32,riscv64))]
fn riscv_m_ptr(_f: extern "riscv-interrupt-m" fn(u8) -> u8) {
}

#[cfg(any(riscv32,riscv64))]
fn riscv_s_ptr(_f: extern "riscv-interrupt-s" fn(u8) -> u8) {
}

#[cfg(any(x64,i686))]
fn x86_ptr(_f: extern "x86-interrupt" fn() -> u8) {
}
