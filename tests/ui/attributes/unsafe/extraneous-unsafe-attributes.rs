//@ edition: 2024
//@ compile-flags: -Zunstable-options
#![feature(unsafe_attributes)]

#[unsafe(cfg(any()))] //~ ERROR: is not an unsafe attribute
fn a() {}

#[unsafe(cfg_attr(any(), allow(dead_code)))] //~ ERROR: is not an unsafe attribute
fn b() {}

#[unsafe(test)] //~ ERROR: is not an unsafe attribute
fn aa() {}

#[unsafe(ignore = "test")] //~ ERROR: is not an unsafe attribute
fn bb() {}

#[unsafe(should_panic(expected = "test"))] //~ ERROR: is not an unsafe attribute
fn cc() {}

#[unsafe(macro_use)] //~ ERROR: is not an unsafe attribute
mod inner {
    #[unsafe(macro_export)] //~ ERROR: is not an unsafe attribute
    macro_rules! m {
        () => {};
    }
}

#[unsafe(used)] //~ ERROR: is not an unsafe attribute
static FOO: usize = 0;

fn main() {}
