//@ run-pass
//@ compile-flags: -Zcontract-checks=yes

#![expect(incomplete_features)]
#![feature(contracts)]
extern crate core;
use core::contracts::requires;

// Compound statements (those using [ExpressionWithBlock]
// (https://doc.rust-lang.org/beta/reference/expressions.html#railroad-ExpressionWithBlock))
// like blocks, if-expressions, and loops require no trailing semicolon. This
// regression test captures the case where the last statement in the contract
// declarations has no trailing semicolon.
#[requires(
    {}
    true
)]
fn foo() {}

fn main() {
    foo()
}
