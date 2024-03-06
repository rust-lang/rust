// Ensure that we don't get a mismatch error when inserting the host param
// at the end of generic args when the generics have defaulted params.
//
//@ check-pass

#![feature(const_trait_impl, effects)]

#[const_trait]
pub trait Foo<Rhs: ?Sized = Self> {
    /* stuff */
}

impl const Foo for () {}

fn main() {}
