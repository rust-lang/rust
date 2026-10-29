// Test for #120759, deadlock detected without any query

#![crate_type = "lib"]
#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    //~^ ERROR unresolved import `std::mem::BikeshedIntrinsicFrom`
    pub struct Context;

    pub fn is_maybe_transmutable<Src, Dst>(&self, cpu: &mut CPU)
    //~^ ERROR `self` parameter is only allowed in associated functions
    //~| ERROR cannot find type `CPU` in this scope
    where
        Dst: BikeshedIntrinsicFrom<Src, Context>,
    {
    }
}

fn should_pad_explicitly_packed_field() {
    #[repr(C)]
    struct ExplicitlyPadded(ExplicitlyPadded);
    //~^ ERROR recursive type `ExplicitlyPadded` has infinite size

    assert::is_maybe_transmutable::<ExplicitlyPadded, ()>();
}
