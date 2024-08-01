//@ build-pass
#![feature(unsafe_attributes)]

#[unsafe(no_mangle)]
fn a() {}

fn main() {}
