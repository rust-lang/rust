//@ run-pass

#![feature(ptr_internals, test)]

extern crate test;
use test::black_box as b; // prevent promotion of the argument and const-propagation of the result

use std::ptr::Unique;


const PTR: *mut u32 = Unique::dangling().as_ptr();

pub fn main() {
    // Be super-extra paranoid and cast the fn items to fn pointers before blackboxing them.
    assert_eq!(PTR, b::<fn() -> _>(Unique::<u32>::dangling)().as_ptr());
}
