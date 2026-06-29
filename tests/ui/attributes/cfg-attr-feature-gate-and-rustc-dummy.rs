//! Regression test for https://github.com/rust-lang/rust/issues/24434

//@ check-pass

#![cfg_attr(true, feature(rustc_attrs))]
#![rustc_dummy]

fn main() {}
