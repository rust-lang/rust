// check-pass
#![feature(const_raw_ptr_deref)]
#![feature(raw_ref_macros)]

use std::ptr;

const fn test_fn(x: *const i32) {
    let x2 = unsafe { ptr::raw_const!(*x) };
}

fn main() {}
