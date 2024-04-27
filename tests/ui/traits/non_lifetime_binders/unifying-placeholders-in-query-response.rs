//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

pub trait Foo<T: ?Sized> {
    type Bar<K: ?Sized>: ?Sized;
}

impl Foo<usize> for () {
    type Bar<K: ?Sized> = K;
}

pub fn f<T1, T2>(a: T1, b: T2)
where
    T1: for<T> Foo<usize, Bar<T> = T>,
    T2: for<T> Foo<usize, Bar<T> = <T1 as Foo<usize>>::Bar<T>>,
{
}

fn it_works() {
    f((), ());
}

fn main() {}
