#!/usr/bin/env -S cargo +nightly -Zscript
// Make sure that shebangs are still allowed even when `-Zcrate-attr` is present.
//@ check-pass
//@ compile-flags: -Zcrate-attr=feature(rustc_attrs)
#[rustc_dummy]
fn main() {}
