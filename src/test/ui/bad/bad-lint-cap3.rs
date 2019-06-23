// check-pass
// compile-flags: --cap-lints warn

#![warn(unused)]
#![deny(warnings)]

use std::option; //~ WARN

fn main() {}
