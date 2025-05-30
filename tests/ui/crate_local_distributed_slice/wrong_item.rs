#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
fn prr() {}
//~^ ERROR expected this to be a const or a static

fn main() {}
