//@ compile-flags: -Znext-solver

// `TransmuteFrom` is not safetly-transmutable.

#![feature(unsafe_binders, transmutability)]
#![allow(incomplete_features)]

use std::mem::TransmuteFrom;

fn is_transmutable<Src, Dst: TransmuteFrom<Src>>() {}

fn main() {
    is_transmutable::<u32, unsafe<> u32>();
    //~^ ERROR `u32` cannot be safely transmuted into `unsafe<> u32`
    is_transmutable::<unsafe<> u32, u32>();
    //~^ ERROR `unsafe<> u32` cannot be safely transmuted into `u32`
}
