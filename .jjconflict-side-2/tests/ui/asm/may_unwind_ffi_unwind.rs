// Check that asm!() with options(may_unwind) is considered an FFI call by has_ffi_unwind_calls.

//@ check-fail
//@ needs-asm-support
//@ needs-unwind

#![feature(asm_unwind)]
#![deny(ffi_unwind_calls)]

use std::arch::asm;

#[no_mangle]
pub unsafe fn asm_may_unwind() {
    asm!("", options(may_unwind));
    //~^ ERROR call to inline assembly that may unwind
}

fn main() {}
