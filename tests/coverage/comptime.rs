#![feature(rustc_attrs)]
//@ edition: 2024

// Check that instrumenting a crate with a comptime function doesn't ICE.
// (The function itself doesn't need to be instrumented, and probably shouldn't be.)

#[rustc_comptime]
fn comptime_fn() {}

fn main() {}
