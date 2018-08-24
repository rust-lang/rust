// compile-flags: --cap-lints warn

#![warn(unused)]
#![deny(warnings)]
#![feature(rustc_attrs)]

use std::option; //~ WARN

#[rustc_error]
fn main() {} //~ ERROR: compilation successful

