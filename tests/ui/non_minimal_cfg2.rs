//@require-annotations-for-level: WARN
#![allow(unused)]

#[cfg(all())]
//~^ ERROR: unneeded sub `cfg` when there is no condition
//~| NOTE: `-D clippy::non-minimal-cfg` implied by `-D warnings`
fn all() {}

fn main() {}
