// This should fail even without Stacked Borrows.
//@compile-flags: -Zmiri-disable-stacked-borrows -Cdebug-assertions=no

#![allow(invalid_reference_casting)] // for u16 -> u32

fn main() {
    // Try many times as this might work by chance.
    for _ in 0..20 {
        let x = [2u16, 3, 4]; // Make it big enough so we don't get an out-of-bounds error.
        let x = &x[0] as *const _ as *const u32;
        // This must fail because alignment is violated: the allocation's base is not sufficiently aligned.
        let _x = unsafe { &*x }; //~ ERROR: required 4 byte alignment
    }
}
