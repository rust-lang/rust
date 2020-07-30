// Checks that lifetimes cannot be interspersed between consts and types.

#![feature(const_generics)]
#![allow(incomplete_features)]

struct Foo<const N: usize, 'a, T = u32>(&'a (), T);
//~^ Error lifetime parameters must be declared prior to const parameters

fn main() {}
