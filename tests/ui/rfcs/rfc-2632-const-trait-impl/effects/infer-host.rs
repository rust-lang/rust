// check-pass

#![feature(const_trait_impl, effects)]

// Host param needs to get infer even if there are generic params supplied

pub const fn hmm<T>() -> usize {
    1
}

pub const fn uwu(x: [u8; hmm::<()>()]) {}

fn main() {}
