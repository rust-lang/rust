//! regression test for <https://github.com/rust-lang/rust/issues/42453>
//! struct named "builder" conflicted with derive macro internals.
//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

#[derive(Debug)]
struct builder;

fn main() {}
