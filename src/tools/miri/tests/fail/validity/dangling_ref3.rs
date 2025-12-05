// Make sure we catch this even without Stacked Borrows
//@compile-flags: -Zmiri-disable-stacked-borrows

#![allow(dangling_pointers_from_locals)]

use std::mem;

fn dangling() -> *const u8 {
    let x = 0u8;
    &x as *const _
}

fn main() {
    let _x: &i32 = unsafe { mem::transmute(dangling()) }; //~ ERROR: dangling reference (use-after-free)
}
