//@ known-bug: #111419
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub trait Example<const X: usize, const Y: usize, const Z: usize = { X + Y }>
where
    [(); X + Y]:,
{}

impl<const X: usize, const Y: usize> Example<X, Y> for Value {}

pub struct Value;

fn main() {}
