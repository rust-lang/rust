//@ compile-flags: --crate-type=lib

// Regression test for <https://github.com/rust-lang/rust/issues/120033>

#![feature(rustc_attrs)]
#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]
#![rustc_no_implicit_bounds]

pub trait Foo<T> {
    type Bar<K>;
}

pub struct Bar<T: ?AutoTrait> {} //~ ERROR cannot find trait `AutoTrait`

pub fn f<T1, T2>()
where
    T1: for<T> Foo<usize, Bar = Bar<T>>, //~ ERROR missing generics for associated type `Foo::Bar`
    //~| ERROR missing generics for associated type `Foo::Bar`
    T2: for<L, T> Foo<usize, Bar<T> = T1::Bar<T>>,
{
}
