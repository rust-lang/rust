// Checks that const expressions have a useful note explaining why they can't be evaluated.
// The note should relate to the fact that it cannot be shown forall N that it maps 1-1 to a new
// type.

#![feature(const_generics)]
#![allow(incomplete_features)]

struct Collatz<const N: Option<usize>>;

impl <const N: usize> Collatz<{Some(N)}> {}
//~^ ERROR the const parameter

struct Foo;

impl<const N: usize> Foo {}
//~^ ERROR the const parameter

fn main() {}
