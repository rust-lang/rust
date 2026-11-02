//! This UI test ensures that transmutability check tests nested goals.
//! If the nested goals regarding the NewType implementing the trait
//! was not being tested, this transmute call would compile.
//
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ check-fail

#![feature(transmutability)]
use std::mem::TransmuteFrom;

struct NewType(u8);

fn main() {
    let bytes: &[NewType; 4] = unsafe { TransmuteFrom::<&i32>::transmute(&1i32) };
    //[current]~^ ERROR `i32` cannot be safely transmuted into `[NewType; 4]`
    //[next]~^^ ERROR `&i32` cannot be safely transmuted into `&[NewType; 4]`
}
