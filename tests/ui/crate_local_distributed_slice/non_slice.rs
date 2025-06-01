#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
const MEOWS: &str;
//~^ ERROR expected this type to be an array

distributed_slice_element!(MEOWS, "hellow");

fn main() {}
