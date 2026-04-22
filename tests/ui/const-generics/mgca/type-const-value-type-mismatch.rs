// Regression test for https://github.com/rust-lang/rust/issues/152962

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ compile-flags: -Zvalidate-mir

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

pub struct A;

pub trait Array {
    type const LEN: usize;
    fn arr() -> [u8; Self::LEN];
}

impl Array for A {
    type const LEN: usize = 0u8;
    //~^ ERROR the constant `0` is not of type `usize`

    fn arr() -> [u8; const { Self::LEN }] {}
    //~^ ERROR mismatched types [E0308]
    //[current]~| ERROR method `arr` has an incompatible type for trait [E0053]
    //[next]~| ERROR type annotations needed
    //[next]~| ERROR type mismatch resolving `<A as Array>::LEN normalizes-to 0` [E0271]
}

fn main() {}
