//@ check-pass
//@ compile-flags: --emit=mir
#![feature(rustc_attrs)]

#[rustc_comptime]
fn comptime_fn() {}

fn main() {}
