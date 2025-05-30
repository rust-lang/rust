#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
const MEOWS: [&str; _];
//~^ ERROR cycle detected when simplifying constant for the type system `MEOWS`

distributed_slice_element!(MEOWS, "mrow");
distributed_slice_element!(MEOWS, MEOWS[0]);

fn main() {}
