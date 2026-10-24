// Alias might not cover type parameters.

//@ revisions: classic next
//@[next] compile-flags: -Znext-solver

//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021

//@ known-bug: #99554
//@ check-pass

trait Id {
    type Assoc;
}

impl<T> Id for T {
    type Assoc = T;
}

pub struct B;
impl<T> foreign::Trait2<B, T> for <T as Id>::Assoc {
    type Assoc = usize;
}

fn main() {}
