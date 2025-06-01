#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
const MEOWS: [&str; _];

distributed_slice_element!(MEOWS, 10);
//~^ ERROR mismatched types

fn main() {}
