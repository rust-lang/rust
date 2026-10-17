#![deny(unused_attributes)]

#[path = "auxiliary/submod.rs"]
mod submod;

fn main() {}

//~? ERROR the `#![crate_name]` attribute can only be used at the crate root
