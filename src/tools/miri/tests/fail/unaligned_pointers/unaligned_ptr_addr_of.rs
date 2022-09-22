// This should fail even without validation or Stacked Borrows.
//@compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows
use std::ptr;

fn main() {
    // Try many times as this might work by chance.
    for _ in 0..10 {
        let x = [2u16, 3, 4]; // Make it big enough so we don't get an out-of-bounds error.
        let x = &x[0] as *const _ as *const u32;
        // This must fail because alignment is violated: the allocation's base is not sufficiently aligned.
        // The deref is UB even if we just put the result into a raw pointer.
        let _x = unsafe { ptr::addr_of!(*x) }; //~ ERROR: memory with alignment 2, but alignment 4 is required
    }
}
