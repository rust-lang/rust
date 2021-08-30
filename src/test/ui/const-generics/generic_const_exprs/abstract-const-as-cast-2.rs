#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Evaluatable<const N: u128> {}

struct Foo<const N: u8>([u8; N as usize])
//~^ Error: unconstrained generic constant
//~| help: try adding a `where` bound using this expression: `where [(); N as usize]:`
where
    Evaluatable<{N as u128}>:;

struct Foo2<const N: u8>(Evaluatable::<{N as u128}>) where Evaluatable<{N as usize as u128 }>:;
//~^ Error: unconstrained generic constant
//~| help: try adding a `where` bound using this expression: `where [(); {N as u128}]:`

struct Bar<const N: u8>([u8; (N + 2) as usize]) where [(); (N + 1) as usize]:;
//~^ Error: unconstrained generic constant
//~| help: try adding a `where` bound using this expression: `where [(); (N + 2) as usize]:`

fn main() {}
