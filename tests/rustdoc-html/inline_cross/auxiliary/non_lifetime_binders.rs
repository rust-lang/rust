#![feature(non_lifetime_binders)]

pub trait Trait<T> {}

pub fn f(_: impl for<T> Trait<T>) {}

pub fn g<T>(_: T)
where
    T: for<U> Trait<U>,
{}
