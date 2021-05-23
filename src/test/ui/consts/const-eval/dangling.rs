#![feature(const_raw_ptr_deref)]

use std::mem;

// Make sure we error with the right kind of error on a too large slice.
const TEST: () = { unsafe {
    let slice: *const [u8] = mem::transmute((1usize, usize::MAX));
    let _val = &*slice; //~ ERROR: any use of this value will cause an error
    //~| slice is bigger than largest supported object
    //~| WARN this was previously accepted by the compiler but is being phased out
} };

fn main() {}
