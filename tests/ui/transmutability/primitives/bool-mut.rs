// check-fail
//[next] compile-flags: -Ztrait-solver=next

#![feature(transmutability)]
mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, { Assume::SAFETY }>
    {}
}

fn main() {
    assert::is_transmutable::<&'static mut bool, &'static mut u8>() //~ ERROR cannot be safely transmuted
}
