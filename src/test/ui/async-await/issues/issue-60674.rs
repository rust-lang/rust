// aux-build:issue-60674.rs
// build-pass (FIXME(62277): could be check-pass?)
// edition:2018
#![feature(async_await)]

// This is a regression test that ensures that `mut` patterns are not lost when provided as input
// to a proc macro.

extern crate issue_60674;

#[issue_60674::attr]
async fn f(mut x: u8) {}

#[issue_60674::attr]
async fn g((mut x, y, mut z): (u8, u8, u8)) {}

#[issue_60674::attr]
async fn g(mut x: u8, (a, mut b, c): (u8, u8, u8), y: u8) {}

fn main() {}
