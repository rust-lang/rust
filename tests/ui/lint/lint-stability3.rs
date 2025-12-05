//@ aux-build:lint_stability.rs

#![deny(deprecated)]
#![allow(warnings)]

#[macro_use]
extern crate lint_stability;

use lint_stability::*;

fn main() {
    macro_test_arg_nested!(deprecated_text);
    //~^ ERROR use of deprecated function `lint_stability::deprecated_text`: text
}
