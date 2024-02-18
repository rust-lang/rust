//@ compile-flags: --cap-lints deny

#![warn(unused)]
#![deny(warnings)]

use std::option; //~ ERROR

fn main() {}
