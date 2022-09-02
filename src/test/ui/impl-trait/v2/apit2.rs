// check-pass

#![feature(return_position_impl_trait_v2)]

fn i<T>(a: impl Iterator<Item = T>) -> impl Iterator<Item = T> {
    a
}

fn main() {}
