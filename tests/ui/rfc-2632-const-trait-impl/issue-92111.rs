// Regression test for #92111.
//
// check-pass

#![feature(const_trait_impl)]

use std::marker::Destruct;

pub trait Tr {}

#[allow(drop_bounds)]
impl<T: Drop> Tr for T {}

#[derive(Debug)]
pub struct S(i32);

impl Tr for S {}

const fn a<T: ~const Destruct>(t: T) {}

fn main() {
    a(S(0));
}
