/*! Tests interrupt ABIs can return !

Most interrupt ABIs share a similar restriction in terms of not allowing most signatures,
but it makes sense to allow them to return ! because they could indeed be divergent.

This test uses `cfg` because it is not testing whether these ABIs work on the platform.
*/
//@ add-core-stubs
//@ revisions: x64 i686 riscv32 riscv64 avr msp430
//@ build-pass
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

/* interrupts can return never */

#[cfg(avr)]
extern "avr-interrupt" fn avr_ret_never() -> ! {
    loop {}
}

#[cfg(msp430)]
extern "msp430-interrupt" fn msp430_ret_never() -> ! {
    loop {}
}

#[cfg(any(riscv32,riscv64))]
extern "riscv-interrupt-m" fn riscv_m_ret_never() -> ! {
    loop {}
}

#[cfg(any(riscv32,riscv64))]
extern "riscv-interrupt-s" fn riscv_s_ret_never() -> ! {
    loop {}
}

#[cfg(any(x64,i686))]
extern "x86-interrupt" fn x86_ret_never(_p: *const u8) -> ! {
    loop {}
}

/* interrupts can return explicit () */

#[cfg(avr)]
extern "avr-interrupt" fn avr_ret_unit() -> () {
    ()
}

#[cfg(msp430)]
extern "msp430-interrupt" fn msp430_ret_unit() -> () {
    ()
}

#[cfg(any(riscv32,riscv64))]
extern "riscv-interrupt-m" fn riscv_m_ret_unit() -> () {
    ()
}

#[cfg(any(riscv32,riscv64))]
extern "riscv-interrupt-s" fn riscv_s_ret_unit() -> () {
    ()
}

#[cfg(any(x64,i686))]
extern "x86-interrupt" fn x86_ret_unit(_x: *const u8) -> () {
    ()
}

/* extern "interrupt" fnptrs can return ! too */

#[cfg(avr)]
fn avr_ptr(_f: extern "avr-interrupt" fn() -> !) {
}

#[cfg(msp430)]
fn msp430_ptr(_f: extern "msp430-interrupt" fn() -> !) {
}

#[cfg(any(riscv32,riscv64))]
fn riscv_m_ptr(_f: extern "riscv-interrupt-m" fn() -> !) {
}

#[cfg(any(riscv32,riscv64))]
fn riscv_s_ptr(_f: extern "riscv-interrupt-s" fn() -> !) {
}

#[cfg(any(x64,i686))]
fn x86_ptr(_f: extern "x86-interrupt" fn() -> !) {
}
