//! Regression test for an ICE in borrowck when a derived `Reborrow` impl is rejected because the
//! type has no lifetime parameter. The issue's own reproducer no longer reaches borrowck (its
//! features were renamed and its `{ (1, _) }` argument is now rejected earlier), so this is a
//! reduced variant that still hits the same ICE.
//! See <https://github.com/rust-lang/rust/issues/162108>.

#![feature(reborrow)]

use std::marker::Reborrow;

#[derive(Reborrow)]
//~^ ERROR implementing `Reborrow` requires that a single lifetime parameter is passed between source and target
struct S<const X: u32>;

fn main() {
    let _: S<1> = S::<1>;
}
