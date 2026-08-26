//@ compile-flags: -Cinstrument-coverage
//@ needs-profiler-runtime
//@ build-pass

// Check that we don't try to generate coverage data for compile-time-only functions
// Regression test for https://github.com/rust-lang/rust/pull/161808

#![feature(rustc_attrs)]

#[expect(unused)]
#[rustc_comptime]
fn comptime_fn() {}

fn main() {}
