// run-pass

#![feature(ptr_internals, test)]

extern crate test;
use test::black_box as b; // prevent promotion of the argument and const-propagation of the result

use std::ptr::NonNull;

const DANGLING: NonNull<u32> = NonNull::dangling();
const CASTED: NonNull<u32> = NonNull::cast(NonNull::<i32>::dangling());

pub fn main() {
    assert_eq!(DANGLING, b(NonNull::dangling()));
    assert_eq!(CASTED, b(NonNull::dangling()));
}
