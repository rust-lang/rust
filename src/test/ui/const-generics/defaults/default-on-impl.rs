// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]

struct Foo<const N: usize>;

impl<const N: usize = 1> Foo<N> {}
//~^ ERROR defaults for const parameters are only allowed

fn main() {}
