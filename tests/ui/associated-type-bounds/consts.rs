#![feature(associated_type_bounds)]

pub fn accept(_: impl Trait<K: Copy>) {}
//~^ ERROR expected associated type, found associated constant

pub trait Trait {
    const K: i32;
}

fn main() {}
