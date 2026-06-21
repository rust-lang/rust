#![expect(incomplete_features)]
#![feature(min_generic_const_args)]


type const A: u8 = A;
//~^ ERROR: overflow normalizing the alias const `A` [E0275]

fn main() {}
