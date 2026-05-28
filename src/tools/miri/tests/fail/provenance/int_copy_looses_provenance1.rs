use std::mem;

// Doing a copy at integer type should lose provenance.
// This tests the optimized-array case of integer copies.
fn main() {
    let ptrs = [&42];
    let ints: [usize; 1] = unsafe { mem::transmute(ptrs) };
    let ptr = (&raw const ints[0]).cast::<&i32>();
    let _val = unsafe { *ptr.read() }; //~ERROR: dangling
}
