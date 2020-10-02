#![feature(min_const_generics)]

fn foo<const A: usize, const B: usize>(bar: [usize; A + B]) {}
//~^ ERROR generic parameters must not be used inside of non-trivial constant values
//~| ERROR generic parameters must not be used inside of non-trivial constant values

fn main() {}
