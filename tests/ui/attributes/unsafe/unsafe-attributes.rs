//@ build-pass
#![feature(unsafe_attributes)]

#[unsafe(no_mangle)]
fn a() {}

#[unsafe(export_name = "foo")]
fn b() {}

#[unsafe(link_section = ".example_section")]
static VAR1: u32 = 1;

#[cfg_attr(any(), unsafe(no_mangle))]
static VAR2: u32 = 1;

fn main() {}
