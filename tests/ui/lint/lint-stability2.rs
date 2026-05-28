//@ aux-build:lint_stability.rs

#![deny(deprecated)]

#[macro_use]
extern crate lint_stability;

use lint_stability::*;

fn main() {
    macro_test!(); //~ ERROR use of deprecated function `lint_stability::deprecated`: text
}
