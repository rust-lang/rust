//@ run-rustfix
//@ compile-flags: -Znext-solver
#![feature(gca_min_const_items, gca_const_items, generic_const_items)]
#![expect(incomplete_features)]
#![allow(dead_code)]

use std::gca;

pub trait Trait {
    const N: usize;
    fn foo() -> [u8; Self::N];
    //~^ ERROR generic parameters in const blocks are not allowed
    fn bar() -> [u8; <Self as Trait>::N];
    //~^ ERROR generic parameters in const blocks are not allowed
}

struct S<const N: usize>;

fn baz<T: Trait>() -> [u8; T::N] {
    //~^ ERROR generic parameters in const blocks are not allowed
    todo!()
}

fn qux<T: Trait>() -> S<{ T::N }> {
    //~^ ERROR generic parameters in const blocks are not allowed
    todo!()
}

const ID<const N: usize>: usize = N;

fn quux<const N: usize>() -> [u8; ID::<N>] {
    //~^ ERROR generic parameters in const blocks are not allowed
    todo!()
}

fn main() {}
