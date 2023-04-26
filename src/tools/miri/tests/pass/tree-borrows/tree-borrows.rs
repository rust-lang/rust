//@compile-flags: -Zmiri-tree-borrows

use std::mem;

fn main() {
    aliasing_read_only_mutable_refs();
    string_as_mut_ptr();
}

// Tree Borrows has no issue with several mutable references existing
// at the same time, as long as they are used only immutably.
// I.e. multiple Reserved can coexist.
pub fn aliasing_read_only_mutable_refs() {
    unsafe {
        let base = &mut 42u64;
        let r1 = &mut *(base as *mut u64);
        let r2 = &mut *(base as *mut u64);
        let _l = *r1;
        let _l = *r2;
    }
}

pub fn string_as_mut_ptr() {
    // This errors in Stacked Borrows since as_mut_ptr restricts the provenance,
    // but with Tree Borrows it should work.
    unsafe {
        let mut s = String::from("hello");
        s.reserve(1); // make the `str` that `s` derefs to not cover the entire `s`.

        // Prevent automatically dropping the String's data
        let mut s = mem::ManuallyDrop::new(s);

        let ptr = s.as_mut_ptr();
        let len = s.len();
        let capacity = s.capacity();

        let s = String::from_raw_parts(ptr, len, capacity);

        assert_eq!(String::from("hello"), s);
    }
}
