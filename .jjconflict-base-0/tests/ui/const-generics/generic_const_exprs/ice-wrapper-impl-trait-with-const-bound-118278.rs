//! Regression test for https://github.com/rust-lang/rust/issues/118278

//@ check-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub trait Foo {
    const SIZE: usize;
}

impl Foo for u64 {
    const SIZE: usize = 8;
}

pub struct Wrapper<T>
where
    T: Foo,
    [(); T::SIZE]:,
{
    pub t: T,
}

pub fn bar() -> Wrapper<impl Foo> {
    Wrapper { t: 10 }
}

fn main() {}
