// compile-flags: -F unused_features
// aux-build:lint_output_format.rs

#![allow(deprecated)]

extern crate lint_output_format; //~ ERROR use of unstable library feature
use lint_output_format::{foo, bar}; //~ ERROR use of unstable library feature

fn main() {
    let _x = foo();
    let _y = bar(); //~ ERROR use of unstable library feature
}
