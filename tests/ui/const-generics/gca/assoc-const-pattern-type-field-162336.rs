//! Regression test for <https://github.com/rust-lang/rust/issues/162336>.
//@ build-pass
//@ compile-flags: -Znext-solver

#![feature(gca_min_const_items, gca_const_items)]

use std::time::Duration;

trait Trait {
    const CONST: Duration;
}

struct Thing;

impl Trait for Thing {
    const CONST: Duration = Duration::ZERO;
}

fn generic<T: Trait>() -> Duration {
    T::CONST
}

fn main() {
    let _ = <Thing as Trait>::CONST;
    let _ = generic::<Thing>();
}
