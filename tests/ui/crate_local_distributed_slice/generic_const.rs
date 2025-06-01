#![allow(incomplete_features)]
#![feature(crate_local_distributed_slice)]
#![feature(generic_const_items)]

#[distributed_slice(crate)]
const MEOWS<T>: [T; _];
//~^ ERROR distributed slices can't be generic

fn main() {}
