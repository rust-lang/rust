// compile-flags: --cap-lints warn

#![warn(unused)]
#![deny(warnings)]
// compile-pass
// skip-codegen
use std::option; //~ WARN


fn main() {}

