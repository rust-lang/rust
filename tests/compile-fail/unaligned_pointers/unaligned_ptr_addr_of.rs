// This should fail even without validation or Stacked Borrows.
// compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows
#![feature(raw_ref_macros)]
use std::ptr;

fn main() {
    let x = [2u16, 3, 4]; // Make it big enough so we don't get an out-of-bounds error.
    let x = &x[0] as *const _ as *const u32;
    // This must fail because alignment is violated: the allocation's base is not sufficiently aligned.
    // The deref is UB even if we just put the result into a raw pointer.
    let _x = unsafe { ptr::raw_const!(*x) }; //~ ERROR memory with alignment 2, but alignment 4 is required
}
