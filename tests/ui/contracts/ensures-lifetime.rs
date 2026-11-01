//@ run-pass
//@ compile-flags: -Zcontract-checks=yes

#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use
//and/or cause compiler crashes [incomplete_features]
#![allow(unused)]

// Regression test to allow contract `ensures` clauses to reference non-static
// references and non-static types. Previously, contracts in the below functions
// would raise type/lifetime errors due a `'static` bound on the `ensures`
// closure.

extern crate core;
use core::contracts::ensures;

#[ensures(|_| { x; true })]
pub fn noop<T>(x: &T) {}

#[ensures(move |_| { x; true })]
pub fn noop_mv<T>(x: &T) {}

#[ensures(|_| { x; true })]
pub fn noop_ptr<T>(x: *const T) {}

#[ensures(move |_| { x; true })]
pub fn noop_ptr_mv<T>(x: *const T) {}

fn main() {}
