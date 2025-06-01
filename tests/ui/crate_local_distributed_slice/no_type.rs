#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
const MEOWS: [_; _];
//~^ ERROR type annotations needed [E0282]

distributed_slice_element!(MEOWS, "MROW");

fn main() {}
