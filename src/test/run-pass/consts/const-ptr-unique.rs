// run-pass

#![feature(ptr_internals, test)]

extern crate test;
use test::black_box as b; // prevent promotion of the argument and const-propagation of the result

use std::ptr::Unique;


const PTR: *mut u32 = Unique::empty().as_ptr();

pub fn main() {
    assert_eq!(PTR, b(Unique::<u32>::empty)().as_ptr());
}
