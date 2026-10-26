//! Regression test for an ICE in borrowck when a `Reborrow` coercion goes through an impl
//! that coherence has already rejected.
//! See <https://github.com/rust-lang/rust/issues/156307>.

#![feature(reborrow)]

use std::marker::Reborrow;

struct Thing;

impl<'a> Reborrow for Thing {}
//~^ ERROR implementing `Reborrow` requires exactly one lifetime argument in the reborrowed type

fn foo(_: Thing) {}

fn main() {
    let x = Thing;
    foo(x);
}
