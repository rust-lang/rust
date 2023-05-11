// check-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub struct Assert<const COND: bool>();
pub trait IsTrue {}
impl IsTrue for Assert<true> {}

pub trait IsNotZST {}
impl<T> IsNotZST for T where Assert<{ std::mem::size_of::<T>() > 0 }>: IsTrue {}

fn main() {}
