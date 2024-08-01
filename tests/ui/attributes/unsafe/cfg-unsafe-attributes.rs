//@ build-pass
#![feature(unsafe_attributes)]

#[cfg_attr(all(), unsafe(no_mangle))]
fn a() {}

fn main() {}
