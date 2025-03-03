//@ check-pass
//@ compile-flags: --crate-type=lib
#![no_std]
#![allow(internal_features)]
#![feature(rustc_attrs, min_specialization)]

// Confirm that specialisation w/ sizedness host effect predicates works.

pub trait Iterator {
    type Item;
}

#[rustc_unsafe_specialization_marker]
pub trait MoreSpecificThanIterator: Iterator {}

pub trait Tr {
    fn foo();
}

impl<A, Iter> Tr for Iter
    where
        Iter: Iterator<Item = A>,
{
    default fn foo() {}
}

impl<A, Iter> Tr for Iter
    where
        Iter: MoreSpecificThanIterator<Item = A>,
{
    fn foo() {}
}
