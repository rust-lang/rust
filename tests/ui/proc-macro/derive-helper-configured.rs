// Derive helpers are resolved successfully inside `cfg_attr`.

//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

#[derive(Empty)]
#[cfg_attr(all(), empty_helper)]
struct S {
    #[cfg_attr(all(), empty_helper)]
    field: u8,
}

fn main() {}
