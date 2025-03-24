//@ check-pass

// This is a regression test for a `span_delayed_bug` during interning when a constant
// evaluates to a (non-dangling) raw pointer.

#![allow(unnecessary_refs)]

const CONST_RAW: *const Vec<i32> = &Vec::new() as *const _;

fn main() {}
