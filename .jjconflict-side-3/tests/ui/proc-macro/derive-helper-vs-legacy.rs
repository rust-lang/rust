//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

#[derive(Empty)]
#[empty_helper] // OK, this is both derive helper and legacy derive helper
#[derive(Empty)]
struct S;

fn main() {}
