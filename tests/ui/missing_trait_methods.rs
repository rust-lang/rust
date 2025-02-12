#![allow(unused, clippy::needless_lifetimes)]
#![warn(clippy::missing_trait_methods)]

trait A {
    fn provided() {}
}

trait B {
    fn required();

    fn a(_: usize) -> usize {
        1
    }

    fn b<'a, T: AsRef<[u8]>>(a: &'a T) -> &'a [u8] {
        a.as_ref()
    }
}

struct Partial;

impl A for Partial {}
//~^ missing_trait_methods

impl B for Partial {
    //~^ missing_trait_methods

    fn required() {}

    fn a(_: usize) -> usize {
        2
    }
}

struct Complete;

impl A for Complete {
    fn provided() {}
}

impl B for Complete {
    fn required() {}

    fn a(_: usize) -> usize {
        2
    }

    fn b<T: AsRef<[u8]>>(a: &T) -> &[u8] {
        a.as_ref()
    }
}

trait MissingMultiple {
    fn one() {}
    fn two() {}
    fn three() {}
}

impl MissingMultiple for Partial {}
//~^ missing_trait_methods
//~| missing_trait_methods
//~| missing_trait_methods

fn main() {}
