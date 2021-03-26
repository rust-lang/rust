// check-pass
#![feature(const_raw_ptr_deref)]

use std::ptr;

const fn test_fn(x: *const i32) {
    let x2 = unsafe { ptr::addr_of!(*x) };
}

fn main() {}
