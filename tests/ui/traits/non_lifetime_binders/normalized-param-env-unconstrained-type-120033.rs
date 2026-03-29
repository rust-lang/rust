//@ compile-flags: --crate-type=lib

// Regression test for <https://github.com/rust-lang/rust/issues/120033>

#![feature(non_lifetime_binders)] //~ WARN the feature `non_lifetime_binders` is incomplete
#![allow(sized_hierarchy_migration)] //~ WARN unknown lint: `sized_hierarchy_migration`
#![feature(sized_hierarchy)] // added to keep parameters unconstrained

pub trait Foo<T: std::marker::PointeeSized> {
    type Bar<K: std::marker::PointeeSized>;
}

pub struct Bar<T: ?AutoTrait> {} //~ ERROR cannot find trait `AutoTrait`

pub fn f<T1, T2>()
where
    T1: for<T> Foo<usize, Bar = Bar<T>>, //~ ERROR missing generics for associated type `Foo::Bar`
    //~| ERROR missing generics for associated type `Foo::Bar`
    T2: for<L, T> Foo<usize, Bar<T> = T1::Bar<T>>,
{
}
