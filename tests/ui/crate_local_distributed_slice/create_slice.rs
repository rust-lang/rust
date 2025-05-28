#![feature(crate_local_distributed_slice)]
// @build-pass

#[distributed_slice(crate)]
const MEOWS: [&str; _];

fn main() {}
