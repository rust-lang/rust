//@ proc-macro: test-macros.rs
//@ proc-macro: extra-empty-derive.rs
//@ build-pass

#[macro_use(Empty)]
extern crate test_macros;
#[macro_use(Empty2)]
extern crate extra_empty_derive;

#[derive(Empty)]
#[empty_helper]
#[derive(Empty2)]
struct S;

fn main() {}
