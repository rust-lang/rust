// edition:2018
// run-pass

// Test that we can use async fns with multiple arbitrary lifetimes.

#![feature(async_await)]
#![allow(dead_code)]

use std::ops::Add;

async fn multiple_hrtb_and_single_named_lifetime_ok<'c>(
    _: impl for<'a> Add<&'a u8>,
    _: impl for<'b> Add<&'b u8>,
    _: &'c u8,
) {}

fn main() {}
