#![feature(const_transmute)]
#![allow(const_err)] // make sure we cannot allow away the errors tested here

use std::mem;

#[derive(Copy, Clone)]
enum Bar {}

union TransmuteUnion<A: Clone + Copy, B: Clone + Copy> {
    a: A,
    b: B,
}

const BAD_BAD_BAD: Bar = unsafe { (TransmuteUnion::<(), Bar> { a: () }).b };
//~^ ERROR it is undefined behavior to use this value

const BAD_BAD_REF: &Bar = unsafe { mem::transmute(1usize) };
//~^ ERROR it is undefined behavior to use this value

const BAD_BAD_ARRAY: [Bar; 1] = unsafe { (TransmuteUnion::<(), [Bar; 1]> { a: () }).b };
//~^ ERROR it is undefined behavior to use this value

fn main() {}
