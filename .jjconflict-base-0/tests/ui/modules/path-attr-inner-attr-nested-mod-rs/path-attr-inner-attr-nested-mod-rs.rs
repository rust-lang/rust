//! Regression test for <https://github.com/rust-lang/rust/issues/162080>
//@ check-pass

#[path = "name/mod.rs"]
mod name;

fn main() {}
