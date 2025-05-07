// Unstable entities should be caught in import lists

//@ aux-build:lint-stability.rs

#![allow(warnings)]

extern crate lint_stability;

use lint_stability::{unstable, deprecated};
//~^ ERROR use of unstable library feature `unstable_test_feature`

fn main() {
}
