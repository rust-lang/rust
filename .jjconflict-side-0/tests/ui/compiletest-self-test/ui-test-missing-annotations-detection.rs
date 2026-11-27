//! Regression test checks UI tests without error annotations are detected as failing.
//!
//! This tests that when we forget to use any `//~ ERROR` comments whatsoever,
//! the test doesn't succeed
//! Originally created in https://github.com/rust-lang/rust/pull/56244

//@ should-fail

fn main() {}
