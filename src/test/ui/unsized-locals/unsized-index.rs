// build-pass (FIXME(62277): could be check-pass?)

// `std::ops::Index` has an `: ?Sized` bound on the `Idx` type param. This is
// an accidental left-over from the times when it `Index` was by-reference.
// Tightening the bound now could be a breaking change. Although no crater
// regression were observed (https://github.com/rust-lang/rust/pull/59527),
// let's be conservative and just add a test for this.
#![feature(unsized_locals)]

use std::ops;

pub struct A;

impl ops::Index<str> for A {
    type Output = ();
    fn index(&self, _: str) -> &Self::Output { panic!() }
}

impl ops::IndexMut<str> for A {
    fn index_mut(&mut self, _: str) -> &mut Self::Output { panic!() }
}

fn main() {}
