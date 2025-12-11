//@ run-pass
//@ compile-flags: -Zcontract-checks=yes

// This test specifically checks that the [incomplete_features] warning is
// emitted when the `contracts` feature gate is enabled, so that it can be
// marked as `expect`ed in other tests in order to reduce duplication.
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]
extern crate core;
use core::contracts::requires;

#[requires(true)]
fn foo() {}

fn main() {
    foo()
}
