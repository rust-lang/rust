// check-pass
#![crate_type = "lib"]
#![feature(transmutability)]
use std::mem::BikeshedIntrinsicFrom;
pub struct Context;

pub fn is_maybe_transmutable<Src, Dst>()
where
    Dst: BikeshedIntrinsicFrom<Src, Context>,
{
}

// The `T` here should not have any effect on checking
// if transmutability is allowed or not.
fn function_with_generic<T>() {
    is_maybe_transmutable::<(), ()>();
}
