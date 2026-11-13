//@ known-bug: #154903
//@ compile-flags: -Zlint-mir
#![feature(guard_patterns)]

fn a(((x if true, _) | (_, x)): (i32, i32)) {}

fn main() {}
