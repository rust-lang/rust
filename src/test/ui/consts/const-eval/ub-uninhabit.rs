#![feature(const_transmute)]

use std::mem;

#[derive(Copy, Clone)]
enum Bar {}

union TransmuteUnion<A: Clone + Copy, B: Clone + Copy> {
    a: A,
    b: B,
}

const BAD_BAD_BAD: Bar = unsafe { (TransmuteUnion::<(), Bar> { a: () }).b };
//~^ ERROR this constant likely exhibits undefined behavior

const BAD_BAD_REF: &Bar = unsafe { mem::transmute(1usize) };
//~^ ERROR this constant likely exhibits undefined behavior

const BAD_BAD_ARRAY: [Bar; 1] = unsafe { (TransmuteUnion::<(), [Bar; 1]> { a: () }).b };
//~^ ERROR this constant likely exhibits undefined behavior

fn main() {}
