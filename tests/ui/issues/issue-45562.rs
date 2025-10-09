//@ run-rustfix

#![deny(unused_attributes)]

#[no_mangle] pub const RAH: usize = 5;
//~^ ERROR const items should never be `#[no_mangle]`

fn main() {}
