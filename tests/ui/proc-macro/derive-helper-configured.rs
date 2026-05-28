// Derive helpers are resolved successfully inside `cfg_attr`.

//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

#[derive(Empty)]
#[cfg_attr(true, empty_helper)]
struct S {
    #[cfg_attr(true, empty_helper)]
    field: u8,
}

fn main() {}
