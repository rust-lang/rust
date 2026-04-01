// Regression test for issue #101980.
// Ensure that we don't suggest impl'ing `Copy` for a type if it already impl's `!Copy`.

//@ check-pass

#![feature(negative_impls)]
#![deny(missing_copy_implementations)]

pub struct Struct {
    pub field: i32,
}

impl !Copy for Struct {}

fn main() {}
