// Test for #120759, which causes an ice bug
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=50 -Zcrate-attr="feature(transmutability)"

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_maybe_transmutable<Src, Dst>(&self, cpu: &mut CPU)
    where
        Dst: BikeshedIntrinsicFrom<Src, Context>,
    {
    }
}

fn should_pad_explicitly_packed_field() {
    #[repr(C)]
    struct ExplicitlyPadded(ExplicitlyPadded);

    assert::is_maybe_transmutable::<ExplicitlyPadded, ()>();
}

fn main() {}
