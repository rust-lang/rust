#![expect(incomplete_features)]
#![feature(associated_const_equality, min_generic_const_args)]

pub fn accept(_: impl Trait<K = 0>) {}

pub trait Trait {
    #[type_const]
    const K: i32;
}
