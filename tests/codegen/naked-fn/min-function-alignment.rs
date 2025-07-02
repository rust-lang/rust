//@ add-core-stubs
//@ compile-flags: -C no-prepopulate-passes -Copt-level=0 -Zmin-function-alignment=16
//@ needs-asm-support
//@ ignore-arm no "ret" mnemonic

#![feature(no_core)]
#![feature(fn_align)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

// functions without explicit alignment use the global minimum
//
// CHECK: .balign 16
#[no_mangle]
#[unsafe(naked)]
pub extern "C" fn naked_no_explicit_align() {
    naked_asm!("ret")
}

// CHECK: .balign 16
#[no_mangle]
#[align(8)]
#[unsafe(naked)]
pub extern "C" fn naked_lower_align() {
    naked_asm!("ret")
}

// CHECK: .balign 32
#[no_mangle]
#[align(32)]
#[unsafe(naked)]
pub extern "C" fn naked_higher_align() {
    naked_asm!("ret")
}

// cold functions follow the same rules as other functions
//
// in GCC, the `-falign-functions` does not apply to cold functions, but
// `-Zmin-function-alignment` applies to all functions.
//
// CHECK: .balign 16
#[no_mangle]
#[cold]
#[unsafe(naked)]
pub extern "C" fn no_explicit_align_cold() {
    naked_asm!("ret")
}
