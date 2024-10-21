// Weak aliases might not cover type parameters.

//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Identity<T> = T;

struct Local;

impl<T> foreign::Trait1<Local, T> for Identity<T> {}
//~^ ERROR type parameter `T` must be covered by another type

fn main() {}
