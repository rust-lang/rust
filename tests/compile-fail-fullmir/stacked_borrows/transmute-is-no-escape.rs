// Make sure we cannot use raw ptrs that got transmuted from mutable references
// (i.e, no EscapeToRaw happened).
// We could, in principle, to EscapeToRaw lazily to allow this code, but that
// would no alleviate the need for EscapeToRaw (see `ref_raw_int_raw` in
// `run-pass/stacked-borrows.rs`), and thus increase overall complexity.
use std::mem;

fn main() {
    let mut x: i32 = 42;
    let raw: *mut i32 = unsafe { mem::transmute(&mut x) };
    unsafe { *raw = 13; } //~ ERROR does not exist on the stack
}
