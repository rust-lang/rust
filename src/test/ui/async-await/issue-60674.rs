// aux-build:issue-60674.rs
// compile-pass
// edition:2018
#![feature(async_await)]

// This is a regression test that ensures that `mut` patterns are not lost when provided as input
// to a proc macro.

extern crate issue_60674;

#[issue_60674::attr]
async fn f(mut x: u8) {}

fn main() {}
