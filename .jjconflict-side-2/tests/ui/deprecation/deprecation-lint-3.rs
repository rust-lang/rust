//@ aux-build:deprecation-lint.rs

#![deny(deprecated)]
#![allow(warnings)]

#[macro_use]
extern crate deprecation_lint;

use deprecation_lint::*;

fn main() {
    macro_test_arg_nested!(deprecated_text);
    //~^ ERROR use of deprecated function `deprecation_lint::deprecated_text`: text
}
