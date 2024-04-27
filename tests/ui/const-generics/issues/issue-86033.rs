//@ check-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait IsTrue<const T: bool> {}
impl IsTrue<true> for () {}

pub trait IsZST {}

impl<T> IsZST for T
where
    (): IsTrue<{ std::mem::size_of::<T>() == 0 }>
{}

fn _func() -> impl IsZST {
    || {}
}

fn main() {}
