//@ run-pass
//@ compile-flags: -Zcontract-checks=yes

#![expect(incomplete_features)]
#![feature(contracts)]

extern crate core;

use core::contracts::{ensures, requires};

pub fn foo() {}

#[requires(foo(); true)]
#[ensures(|_| { true })]
pub const fn bar() {}

fn main() {}
