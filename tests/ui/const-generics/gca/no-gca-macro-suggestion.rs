//@ compile-flags: -Znext-solver
#![feature(gca_min_const_items, gca_const_items, generic_const_items)]
#![expect(incomplete_features)]

use std::gca;

pub trait Trait {
    const N: usize;
    fn foo() -> [u8; Self::N + 1];
    //~^ ERROR generic parameters in const blocks are not allowed
}

struct S<const N: usize>;

fn bar<T: Trait>() -> S<const { T::N }> {
    //~^ ERROR generic parameters in const blocks are not allowed
    todo!()
}

const FOO<const N: usize>: usize = gca!(const { N });
//~^ ERROR generic parameters in const blocks are not allowed

struct W<T>(T);

impl<T> W<T> {
    const M: usize = 1;
    fn baz() -> [u8; Self::M] {
        //~^ ERROR generic `Self` types are currently not permitted in anonymous constants
        todo!()
    }
    fn qux() -> [u8; W::<T>::M] {
        //~^ ERROR generic parameters in const blocks are not allowed
        todo!()
    }
}

macro_rules! len {
    ($t:ty) => { [u8; <$t as Trait>::N] };
}

fn quux<T: Trait>() -> len!(T) {
    //~^ ERROR generic parameters in const blocks are not allowed
    todo!()
}

fn main() {}
