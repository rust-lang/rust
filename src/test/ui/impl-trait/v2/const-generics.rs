// check-pass

#![feature(return_position_impl_trait_v2)]

trait Trait {}
impl Trait for u32 {}

fn rawr<const N: u32>() -> impl Trait {
    0
}

fn main() {}
