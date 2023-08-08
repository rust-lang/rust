#![feature(associated_const_equality)]

pub fn accept(_: impl Trait<K = 0>) {}

pub trait Trait {
    const K: i32;
}
