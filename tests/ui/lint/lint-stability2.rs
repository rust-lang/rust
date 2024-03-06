//@ aux-build:lint_stability.rs
//@ error-pattern: use of deprecated function

#![deny(deprecated)]

#[macro_use]
extern crate lint_stability;

use lint_stability::*;

fn main() {
    macro_test!();
}
