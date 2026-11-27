// Regression test for https://github.com/rust-lang/rust/issues/153005

//@ check-pass

#![feature(non_lifetime_binders)]
#![deny(explicit_outlives_requirements)]

pub struct G<'a, T>
where
    for<V> V: Sized,
{
    pub x: &'a T,
}

fn main() {}
