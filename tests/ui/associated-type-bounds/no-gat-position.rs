#![feature(associated_type_bounds)]

// Test for <https://github.com/rust-lang/rust/issues/119857>.

pub trait Iter {
    type Item<'a>: 'a where Self: 'a;

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a, As1: Copy>>;
    //~^ ERROR associated type bindings are not allowed here
}

impl Iter for () {
    type Item<'a> = &'a mut [()];

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> { None }
}

fn main() {}
