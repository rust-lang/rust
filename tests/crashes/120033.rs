//@ known-bug: #120033
#![feature(non_lifetime_binders)]

pub trait Foo<T: ?Sized> {
    type Bar<K: ?Sized>;
}

pub struct Bar<T: ?AutoTrait> {}

pub fn f<T1, T2>()
where
    T1: for<T> Foo<usize, Bar = Bar<T>>,
    T2: for<L, T> Foo<usize, Bar<T> = T1::Bar<T>>,
{}
