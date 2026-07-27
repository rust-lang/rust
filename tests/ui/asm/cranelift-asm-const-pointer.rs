//@ needs-asm-support
//@ build-pass

// Regression test for Cranelift handling of pointer-valued `asm!` `const` operands

#![feature(asm_const_ptr)]
#![crate_type = "lib"]

use std::arch::asm;
use std::ptr::addr_of;

unsafe extern "C" {
    static FOO: usize;
}

pub fn asm_const_pointer() {
    unsafe {
        asm!("/* {} */", const addr_of!(FOO));
    }
}
