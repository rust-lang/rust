#![crate_type = "lib"]
#![feature(transmutability)]
use std::mem::{Assume, MaybeUninit, TransmuteFrom};

pub fn is_maybe_transmutable<Src, Dst>()
    where Dst: TransmuteFrom<Src, { Assume::VALIDITY.and(Assume::SAFETY) }>
{}

fn extension() {
    is_maybe_transmutable::<(), MaybeUninit<u8>>();
    is_maybe_transmutable::<MaybeUninit<u8>, [u8; 2]>(); //~ ERROR  `MaybeUninit<u8>` cannot be safely transmuted into `[u8; 2]`
}
