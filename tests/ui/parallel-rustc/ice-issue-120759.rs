//@ parallel-front-end
//@ compile-flags: -Z threads=50 -Zcrate-attr="feature(transmutability)"

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    //~^ ERROR
    //~| ERROR
    pub struct Context;

    pub fn is_maybe_transmutable<Src, Dst>(&self, cpu: &mut CPU)
    //~^ ERROR
    //~| ERROR
    where
        Dst: BikeshedIntrinsicFrom<Src, Context>,
    {
    }
}

fn should_pad_explicitly_packed_field() {
    #[repr(C)]
    struct ExplicitlyPadded(ExplicitlyPadded);
    //~^ ERROR

    assert::is_maybe_transmutable::<ExplicitlyPadded, ()>();
}

fn main() {}
