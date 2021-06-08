#![feature(const_evaluatable_checked, const_generics)]
#![allow(incomplete_features)]

trait Evaluatable<const N: u128> {}
impl<const N: u128> Evaluatable<N> for () {}

struct Foo<const N: u8>([u8; N as usize])
//~^ Error: unconstrained generic constant
//~| help: try adding a `where` bound using this expression: `where [(); N as usize]:`
where
    (): Evaluatable<{N as u128}>;

fn main() {}
