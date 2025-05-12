// Ensure that we don't get a mismatch error when inserting the host param
// at the end of generic args when the generics have defaulted params.
//
//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
pub trait Foo<Rhs: ?Sized = Self> {
    /* stuff */
}

impl const Foo for () {}

fn main() {}
