#![deny(unused_attributes)]

mod submod;

fn main() {}

//~? ERROR the `#![crate_name]` attribute can only be used at the crate root
