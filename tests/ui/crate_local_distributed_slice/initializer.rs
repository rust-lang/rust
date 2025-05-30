#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
const MEOWS: [&str; _] = ["meow"];
//~^ ERROR distributed slice elements are added with `distributed_slice_element!(...)`

fn main() {}
