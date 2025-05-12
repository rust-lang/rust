use std::mem;

// Doing a copy at integer type should lose provenance.
// This tests the unoptimized base case.
fn main() {
    let ptrs = [(&42, true)];
    let ints: [(usize, bool); 1] = unsafe { mem::transmute(ptrs) };
    let ptr = (&raw const ints[0].0).cast::<&i32>();
    let _val = unsafe { *ptr.read() }; //~ERROR: dangling
}
