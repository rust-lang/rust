//@ known-bug: unknown
//! Regression test for: https://github.com/rust-lang/rust/issues/144957
//!
//! This test ensures that lifetime information is included in diagnostics.
//!
//! Specifically, it checks that the `become` call produces an error with lifetimes shown
//! in both caller and callee signatures.
//!
//! If the test fails:
//! - Lifetimes may be missing (fix the diagnostic), or
//! - The message format changed (update the test).

#![feature(explicit_tail_calls)]
#![allow(incomplete_features)]

fn foo<'a>(_: fn(&'a ())) {
    become bar(dummy);
}

fn bar(_: fn(&())) {}

fn dummy(_: &()) {}

fn foo_(_: fn(&())) {
    become bar1(dummy2);
}

fn bar1<'a>(_: fn(&'a ())) {}

fn dummy2(_: &()) {}

fn foo__(_: fn(&'static ())) {
    become bar(dummy3);
}

fn bar2(_: fn(&())) {}

fn dummy3(_: &()) {}

fn main() {}
