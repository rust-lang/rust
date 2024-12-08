//@ check-pass
#![allow(dead_code)]
use std::ops::Add;

fn ref_add<T>(a: &T, b: &T) -> T
where
    for<'x> &'x T: Add<&'x T, Output = T>,
{
    a + b
}

fn main() {}
