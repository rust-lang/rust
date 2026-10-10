// Regression test for #106126.
//@ check-pass
//@ aux-build:issue-106126.rs

#![deny(unsafe_op_in_unsafe_fn)]

#[macro_use]
extern crate issue_106126;

foo!();

fn main() {}
