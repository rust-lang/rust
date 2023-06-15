// check-fail
// FIXME(bryangarza): Change to check-pass when coinduction is supported for BikeshedIntrinsicFrom
#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, {
            Assume {
                alignment: true,
                lifetimes: false,
                safety: true,
                validity: false,
            }
        }>
    {}
}

fn main() {
    #[repr(C)] struct A(&'static B);
    #[repr(C)] struct B(&'static A);
    assert::is_maybe_transmutable::<&'static A, &'static B>(); //~ overflow evaluating the requirement
    assert::is_maybe_transmutable::<&'static B, &'static A>();
}
