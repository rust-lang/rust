//! E0228 (lifetime bound for trait object cannot be deduced from context) should not be emitted
//! when an undeclared lifetime bound has been specified.
//!
//! Regression test for https://github.com/rust-lang/rust/issues/152014

fn f(_: std::cell::Ref<'undefined, dyn std::fmt::Debug>) {} //~ ERROR use of undeclared lifetime name `'undefined`
fn main() {}
