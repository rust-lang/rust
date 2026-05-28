//@ check-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub struct ConstDefaultUnstable<const N: usize = { const { 3 } }>;

pub fn main() {}
