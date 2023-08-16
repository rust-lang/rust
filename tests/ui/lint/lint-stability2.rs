//@aux-build:lint_stability.rs
//@error-in-other-file: use of deprecated function

#![deny(deprecated)]

#[macro_use]
extern crate lint_stability;

use lint_stability::*;

fn main() {
    macro_test!();
}
