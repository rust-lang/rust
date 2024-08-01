//@ build-pass
#![feature(unsafe_attributes)]

#[unsafe(no_mangle)]
fn a() {}

#[unsafe(export_name = "foo")]
fn b() {}

#[cfg_attr(any(), unsafe(no_mangle))]
static VAR2: u32 = 1;

fn main() {}
