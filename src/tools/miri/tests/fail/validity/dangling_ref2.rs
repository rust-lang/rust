// Make sure we catch this even without Stacked Borrows
//@compile-flags: -Zmiri-disable-stacked-borrows
use std::mem;

fn main() {
    let val = 14;
    let ptr = (&val as *const i32).wrapping_offset(1);
    let _x: &i32 = unsafe { mem::transmute(ptr) }; //~ ERROR: dangling reference (going beyond the bounds of its allocation)
}
