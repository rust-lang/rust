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

struct A;
struct B;

fn foo<'a>(_: fn(&'a ()), _: A) {
    become bar(dummy, B);
    //~^ ERROR mismatched signatures
    //~| NOTE `become` requires caller and callee to have matching signatures
    //~| NOTE caller signature: `fn(fn(&'a ()), A)`
    //~| NOTE callee signature: `fn(for<'a> fn(&'a ()), B)`
}

fn bar(_: fn(&()), _: B) {}

fn dummy(_: &()) {}

fn foo2(_: fn(&()), _: A) {
    become bar2(dummy2, B);
    //~^ ERROR mismatched signatures
    //~| NOTE `become` requires caller and callee to have matching signatures
    //~| NOTE caller signature: `fn(for<'a> fn(&'a ()), A)`
    //~| NOTE callee signature: `fn(fn(&'a ()), B)`
}

fn bar2<'a>(_: fn(&'a ()), _: B) {}

fn dummy2(_: &()) {}

fn foo3(_: fn(&'static ()), _: A) {
    become bar3(dummy3, B);
    //~^ ERROR mismatched signatures
    //~| NOTE `become` requires caller and callee to have matching signatures
    //~| NOTE caller signature: `fn(fn(&'static ()), A)`
    //~| NOTE callee signature: `fn(for<'a> fn(&'a ()), B)`
}

fn bar3(_: fn(&()), _: B) {}

fn dummy3(_: &()) {}

fn main() {}
