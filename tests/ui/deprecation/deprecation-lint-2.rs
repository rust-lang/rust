//@ aux-build:deprecation-lint.rs

#![deny(deprecated)]

#[macro_use]
extern crate deprecation_lint;

use deprecation_lint::*;

fn main() {
    macro_test!(); //~ ERROR use of deprecated function `deprecation_lint::deprecated`: text
}
