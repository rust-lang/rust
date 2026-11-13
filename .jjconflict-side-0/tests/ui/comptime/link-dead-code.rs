//@ build-pass
//@ compile-flags: -Clink-dead-code
//! Check that we don't try to generate assembly for comptime
//! fns, even when link-dead-code is active.

#![feature(rustc_attrs)]

#[rustc_comptime]
fn f() {}

fn main() {}
