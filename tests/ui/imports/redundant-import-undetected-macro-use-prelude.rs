// This test demonstrates that we currently don't make an effort to detect
// imports made redundant by the `#[macro_use]` prelude.
// See also the discussion in <https://github.com/rust-lang/rust/pull/122954>.

//@ check-pass
//@ aux-build:two_macros.rs
#![deny(unused_imports)]

#[macro_use]
extern crate two_macros;

// This import is actually redundant due to the `#[macro_use]` above.
use two_macros::n;

// We intentionally reference two items from the `#[macro_use]`'d crate because
// if we were to reference only item `n`, we would flag the `#[macro_use]`
// attribute as redundant which would be correct of course.
// That's interesting on its own -- we prefer "blaming" the `#[macro_use]`
// over the import (here, `use two_macros::n`) when it comes to redundancy.
n!();
m!();

fn main() {}
