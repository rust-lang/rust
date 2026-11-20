#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

const N: usize = 1;

pub struct Data {
    _array: [usize; N],
}

pub type Arr = [u8; 2 + 2];

pub trait Tr<const M: usize> {}

pub struct A;

pub const K: usize = 7;

impl Tr<K> for A {}
