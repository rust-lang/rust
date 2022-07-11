// Make sure we catch this even without Stacked Borrows
//@compile-flags: -Zmiri-disable-stacked-borrows
use std::mem;

fn main() {
    let _x: &i32 = unsafe { mem::transmute(16usize) }; //~ ERROR: encountered a dangling reference (address 0x10 is unallocated)
}
