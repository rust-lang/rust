#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// Check that we don't crash when printing inherent projections in diagnostics.

pub struct Carrier<'a>(&'a ());

pub type User = for<'b> fn(Carrier<'b>::Focus<i32>);

impl<'a> Carrier<'a> {
    pub type Focus<T> = &'a mut User; //~ ERROR overflow evaluating associated type
}

fn main() {}
