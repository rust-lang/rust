//@ revisions: ed2015 ed2024
//@ [ed2015] edition:2015
//@ [ed2024] edition:2024
//@ proc-macro: test-macros.rs


#[macro_use]
extern crate test_macros;

#[identity_attr]
mod m {
    pub struct X;

    type A = Y; //~ ERROR cannot find type `Y` in this scope
    // In edition 2015 we'd suggest referring to `crate::Y` as `Y`, which is not a good suggestion,
    // but in other editions we suggest the appropriate `crate::Y`.
}

struct Y;
type A = X; //~ ERROR cannot find type `X` in this scope

#[derive(Copy)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
mod n {}

fn main() {}
