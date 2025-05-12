//@ run-rustfix
#![feature(generic_const_exprs)]
#![allow(incomplete_features, dead_code)]

struct Evaluatable<const N: u128> {}

struct Foo<const N: u8>([u8; N as usize])
//~^ ERROR unconstrained generic constant
where
    Evaluatable<{N as u128}>:;
//~^ HELP try adding a `where` bound

struct Foo2<const N: u8>(Evaluatable::<{N as u128}>) where Evaluatable<{N as usize as u128 }>:;
//~^ ERROR unconstrained generic constant
//~| HELP try adding a `where` bound

struct Bar<const N: u8>([u8; (N + 2) as usize]) where [(); (N + 1) as usize]:;
//~^ ERROR unconstrained generic constant
//~| HELP try adding a `where` bound

fn main() {}
