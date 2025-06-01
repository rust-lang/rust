#![feature(crate_local_distributed_slice)]

const CONST_MEOWS: [&str; 2] = ["meow", "prrr"];
static STATIC_MEOWS: [&str; 2] = ["meow", "prrr"];

distributed_slice_element!(CONST_MEOWS, "mrow");
//~^ ERROR `distributed_slice_element!()` can only add to a distributed slice
distributed_slice_element!(STATIC_MEOWS, "mrow");
//~^ ERROR `distributed_slice_element!()` can only add to a distributed slice

fn main() {}
