// Make sure we cannot use raw ptrs that got transmuted from mutable references
// (i.e, no EscapeToRaw happened).
// We could, in principle, do EscapeToRaw lazily to allow this code, but that
// would no alleviate the need for EscapeToRaw (see `ref_raw_int_raw` in
// `run-pass/stacked-borrows.rs`), and thus increase overall complexity.
use std::mem;

fn main() {
    let mut x: [i32; 2] = [42, 43];
    let _raw: *mut i32 = unsafe { mem::transmute(&mut x[0]) };
    // `raw` still carries a tag, so we get another pointer to the same location that does not carry a tag
    let raw = (&mut x[1] as *mut i32).wrapping_offset(-1);
    unsafe { *raw = 13 }; //~ ERROR: /write access .* tag does not exist in the borrow stack/
}
