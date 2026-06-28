//@ run-pass
//@ compile-flags: -Zcontract-checks=yes

#![expect(incomplete_features)]
#![feature(contracts)]

extern crate core;

use core::contracts::requires;

pub fn foo() {}

#[requires(foo(); true)]
pub const fn bar() {}

fn main() {}
