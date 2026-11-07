//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

#[derive(Clone)]
struct Bar<const A: usize, const B: usize>
where
    [(); A as usize]:,
    [(); B as usize]:,
{}

fn main() {}
