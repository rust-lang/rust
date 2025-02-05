//@ run-rustfix
#![deny(unused_imports)]
#![allow(useless_deprecated)]

// Check that attributes get removed too. See #87973.
#[deprecated]
#[allow(unsafe_code)]
#[cfg(not(FALSE))]
use std::fs;
//~^ ERROR unused import

fn main() {}
