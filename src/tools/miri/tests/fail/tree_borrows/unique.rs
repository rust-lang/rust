//@revisions: default uniq
//@compile-flags: -Zmiri-tree-borrows
//@[uniq]compile-flags: -Zmiri-unique-is-unique

// A pattern that detects if `Unique` is treated as exclusive or not:
// activate the pointer behind a `Unique` then do a read that is parent
// iff `Unique` was specially reborrowed.

#![feature(ptr_internals)]
use core::ptr::Unique;

fn main() {
    let mut data = 0u8;
    let refmut = &mut data;
    let rawptr = refmut as *mut u8;

    unsafe {
        let uniq = Unique::new_unchecked(rawptr);
        *uniq.as_ptr() = 1; // activation
        let _maybe_parent = *rawptr; // maybe becomes Frozen
        *uniq.as_ptr() = 2;
        //~[uniq]^ ERROR: /write access through .* is forbidden/
        let _definitely_parent = data; // definitely Frozen by now
        *uniq.as_ptr() = 3;
        //~[default]^ ERROR: /write access through .* is forbidden/
    }
}
