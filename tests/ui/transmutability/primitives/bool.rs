// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

#![feature(transmutability)]
mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, { Assume::SAFETY }>
    {}

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, { Assume::SAFETY.and(Assume::VALIDITY) }>
    {}
}

fn main() {
    assert::is_transmutable::<u8, bool>(); //~ ERROR cannot be safely transmuted
    assert::is_maybe_transmutable::<u8, bool>();
    assert::is_transmutable::<bool, u8>();
}
