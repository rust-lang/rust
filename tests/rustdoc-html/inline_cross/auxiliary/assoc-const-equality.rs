#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

pub fn accept(_: impl Trait<K = 0>) {}

pub trait Trait {
    #[rustc_always_gca]
    const K: i32;
}
