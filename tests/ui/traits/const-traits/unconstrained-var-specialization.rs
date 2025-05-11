//@ check-pass
//@ compile-flags: --crate-type=lib
#![no_std]
#![allow(internal_features)]
#![feature(rustc_attrs, min_specialization, const_trait_impl)]

// In the default impl below, `A` is constrained by the projection predicate, and if the host effect
// predicate for `const Foo` doesn't resolve vars, then specialization will fail.

#[const_trait]
trait Foo {}

pub trait Iterator {
    type Item;
}

#[rustc_unsafe_specialization_marker]
pub trait MoreSpecificThanIterator: Iterator {}

pub trait Tr {
    fn foo();
}

impl<A: const Foo, Iter> Tr for Iter
    where
        Iter: Iterator<Item = A>,
{
    default fn foo() {}
}

impl<A: const Foo, Iter> Tr for Iter
    where
        Iter: MoreSpecificThanIterator<Item = A>,
{
    fn foo() {}
}
