#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
const MEOWS: [&str; 10];
//~^ ERROR expected this length to be `_`

fn main() {}
