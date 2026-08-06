//! Regression test for https://github.com/rust-lang/rust/issues/36379
//@ check-pass

fn _test() -> impl Default { }

fn main() {}
