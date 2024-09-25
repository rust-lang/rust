#![feature(const_ptr_is_null)]
use std::ptr;

const IS_NULL: () = {
    assert!(ptr::null::<u8>().is_null());
};
const IS_NOT_NULL: () = {
    assert!(!ptr::null::<u8>().wrapping_add(1).is_null());
};

const MAYBE_NULL: () = {
    let x = 15;
    let ptr = &x as *const i32;
    // This one is still unambiguous...
    assert!(!ptr.is_null());
    // but once we shift outside the allocation, we might become null.
    assert!(!ptr.wrapping_sub(512).is_null()); //~inside `MAYBE_NULL`
};

fn main() {}
