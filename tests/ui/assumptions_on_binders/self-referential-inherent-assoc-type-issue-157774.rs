//! Regression test for <https://github.com/rust-lang/rust/issues/157774>.
//!
//! A self-referential inherent associated type used to ICE with
//! `item_bounds called on` an impl item.

//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

struct Foo<'a> {
    x: &'a (),
}

impl<'a> Foo<'a> {
    type Assoc = &'a Foo<'a>::Assoc;
    //~^ ERROR overflow evaluating the requirement `Foo<'a>::Assoc == _`
    //~| ERROR overflow evaluating whether `&'a Foo<'a>::Assoc` is well-formed
}

fn main() {}
