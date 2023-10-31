// Test that we get the following hint when trying to use a compiler crate without rustc_driver.
// error-pattern: try adding `extern crate rustc_driver;` at the top level of this crate
// compile-flags: --emit link
// normalize-stderr-test ".*crate .* required.*\n\n" -> ""
// normalize-stderr-test: "aborting due to [0-9]+" -> "aborting due to NUMBER"

#![feature(rustc_private)]

extern crate rustc_serialize;

fn main() {}
