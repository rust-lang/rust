//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "rlib"]
#![allow(asm_sub_register)]

use std::arch::asm;
use std::mem::MaybeUninit;

// CHECK-LABEL: @int
#[no_mangle]
pub unsafe fn int(x: MaybeUninit<i32>) -> MaybeUninit<i32> {
    let y: MaybeUninit<i32>;
    asm!("/*{}{}*/", in(reg) x, out(reg) y);
    y
}

// CHECK-LABEL: @inout
#[no_mangle]
pub unsafe fn inout(mut x: i32) -> MaybeUninit<u32> {
    let mut y: MaybeUninit<u32>;
    asm!("/*{}*/", inout(reg) x => y);
    asm!("/*{}*/", inout(reg) y => x);
    asm!("/*{}*/", inlateout(reg) x => y);
    asm!("/*{}*/", inlateout(reg) y => x);
    y
}
