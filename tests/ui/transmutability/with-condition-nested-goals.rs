//@ compile-flags: -Znext-solver=globally
//@ check-fail

#![feature(transmutability)]
use std::mem::TransmuteFrom;

struct NewType(u8);

fn main() {
    let bytes: &[NewType; 4] = unsafe { TransmuteFrom::<&i32>::transmute(&1i32) };
    //~^ ERROR `&i32` cannot be safely transmuted into `&[NewType; 4]`
}
