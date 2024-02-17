// Ensure inout asm! operands are marked as used by the liveness pass

//@ only-x86_64
//@ check-pass

#![allow(dead_code)]
#![warn(unused_assignments)]
#![warn(unused_variables)]

use std::arch::asm;

// Test the single inout case
unsafe fn f1(mut src: *const u8) {
    asm!("/*{0}*/", inout(reg) src); //~ WARN value assigned to `src` is never read
}

unsafe fn f2(mut src: *const u8) -> *const u8 {
    asm!("/*{0}*/", inout(reg) src);
    src
}

// Test the split inout case
unsafe fn f3(mut src: *const u8) {
    asm!("/*{0}*/", inout(reg) src => src); //~ WARN value assigned to `src` is never read
}

unsafe fn f4(mut src: *const u8) -> *const u8 {
    asm!("/*{0}*/", inout(reg) src => src);
    src
}

// Tests the use of field projections
struct S {
    field: *mut u8,
}

unsafe fn f5(src: &mut S) {
    asm!("/*{0}*/", inout(reg) src.field);
}

unsafe fn f6(src: &mut S) {
    asm!("/*{0}*/", inout(reg) src.field => src.field);
}

fn main() {}
