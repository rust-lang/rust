#![feature(reborrow)]

use std::marker::Reborrow;

// Regression test: `reborrow_info` must validate ALL data fields,
// not just stop at the first Reborrow field.

struct Bad<'a> {
    first: &'a mut i32,
    second: String, //~ ERROR the trait bound `String: Copy` is not satisfied
}

impl<'a> Reborrow for Bad<'a> {}

fn main() {}
