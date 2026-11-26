//! Regression test for <https://github.com/rust-lang/rust/issues/165653>.
//!
//! Reborrowing a type with two lifetimes and no `&mut` or `Reborrow` field used to
//! hit a `bug!` in borrowck. It now gives the same error as the one lifetime case.

#![feature(reborrow)]

use std::marker::Reborrow;

impl<'a, 'b> Reborrow for DerivedShared<'a, 'b> {}

struct DerivedShared<'a, 'b> {
    a: &'a &'b str,
}

impl<'a, 'b> DerivedShared<'a, 'b> {
    fn clone(&self) -> Self {
        *self
        //~^ ERROR cannot borrow `*self` as mutable, as it is behind a `&` reference
    }
}

fn main() {}
