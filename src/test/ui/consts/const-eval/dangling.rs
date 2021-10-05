use std::mem;

// Make sure we error with the right kind of error on a too large slice.
const TEST: () = { unsafe {
    let slice: *const [u8] = mem::transmute((1usize, usize::MAX));
    let _val = &*slice; //~ ERROR: evaluation of constant value failed
    //~| slice is bigger than largest supported object
} };

fn main() {}
