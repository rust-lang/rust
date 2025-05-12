//@ check-pass
#![crate_type = "lib"]
#![feature(transmutability)]
use std::mem::TransmuteFrom;

pub fn is_maybe_transmutable<Src, Dst>()
where
    Dst: TransmuteFrom<Src>,
{
}

// The `T` here should not have any effect on checking
// if transmutability is allowed or not.
fn function_with_generic<T>() {
    is_maybe_transmutable::<(), ()>();
}
