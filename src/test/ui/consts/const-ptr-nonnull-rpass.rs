// run-pass

#![feature(ptr_internals, test)]

extern crate test;
use test::black_box as b; // prevent promotion of the argument and const-propagation of the result

use std::ptr::NonNull;

const DANGLING: NonNull<u32> = NonNull::dangling();
const CASTED: NonNull<u32> = NonNull::cast(NonNull::<i32>::dangling());

pub fn main() {
    // Be super-extra paranoid and cast the fn items to fn pointers before blackboxing them.
    assert_eq!(DANGLING, b::<fn() -> _>(NonNull::dangling)());
    assert_eq!(CASTED, b::<fn() -> _>(NonNull::dangling)());
}
