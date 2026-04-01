use std::mem;

// Doing a copy at integer type should lose provenance.
// This tests the case where provenacne is hiding in the metadata of a pointer.
fn main() {
    let ptrs = [(&42, &42)];
    // Typed copy at wide pointer type (with integer-typed metadata).
    let ints: [*const [usize]; 1] = unsafe { mem::transmute(ptrs) };
    // Get a pointer to the metadata field.
    let ptr = (&raw const ints[0]).wrapping_byte_add(mem::size_of::<*const ()>()).cast::<&i32>();
    let _val = unsafe { *ptr.read() }; //~ERROR: dangling
}
