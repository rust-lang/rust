//@ known-bug: #120033
#![feature(non_lifetime_binders)]
#![allow(sized_hierarchy_migration)]
#![feature(sized_hierarchy)] // added to keep parameters unconstrained

pub trait Foo<T: std::marker::PointeeSized> {
    type Bar<K: std::marker::PointeeSized>;
}

pub struct Bar<T: ?AutoTrait> {}

pub fn f<T1, T2>()
where
    T1: for<T> Foo<usize, Bar = Bar<T>>,
    T2: for<L, T> Foo<usize, Bar<T> = T1::Bar<T>>,
{}
