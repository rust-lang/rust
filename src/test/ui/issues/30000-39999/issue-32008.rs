// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Tests that binary operators allow subtyping on both the LHS and RHS,
// and as such do not introduce unnecessarily strict lifetime constraints.

use std::ops::Add;

struct Foo;

impl<'a> Add<&'a Foo> for &'a Foo {
    type Output = ();
    fn add(self, rhs: &'a Foo) {}
}

fn try_to_add(input: &Foo) {
    let local = Foo;

    // Manual reborrow worked even with invariant trait search.
    &*input + &local;

    // Direct use of the reference on the LHS requires additional
    // subtyping before searching (invariantly) for `LHS: Add<RHS>`.
    input + &local;
}

fn main() {
}
