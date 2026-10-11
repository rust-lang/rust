#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

pub fn accept(_: impl Trait<K = 0>) {}

pub trait Trait {
    #[rustc_always_gca]
    const K: i32;
}
