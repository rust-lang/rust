// run-pass

#![feature(ptr_internals)]

use std::ptr::Unique;

const fn as_ptr<T>(ptr: Unique<T>) -> *mut T {
    ptr.as_ptr()
}

pub fn main() {
    let mut i: i32 = 10;
    let unique = Unique::new(&mut i).unwrap();
    assert_eq!(unique.as_ptr(), as_ptr(unique));
}
