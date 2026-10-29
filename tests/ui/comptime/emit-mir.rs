//@ check-pass
//@ compile-flags: --emit=mir
#![feature(comptime)]

#[comptime]
fn comptime_fn() {}

fn main() {}
