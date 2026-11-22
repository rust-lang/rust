// Regression test for #106126.
// This used to be check-pass, but had to be changed to check-fail, because
// keeping check-pass requires changing it in a way that stops it from catching
// regressions.
//@ check-fail

//@ aux-build:issue-106126.rs

#![deny(unsafe_op_in_unsafe_fn)]

#[macro_use]
extern crate issue_106126;

foo!();
//~^ ERROR: call to unsafe function `__unsf` is unsafe

fn main() {}
