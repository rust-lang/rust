// compile-flags: --force-warns nonstandard_style -Zunstable-options
// check-pass

#![allow(warnings)]

pub fn FUNCTION() {}
//~^ WARN function `FUNCTION` should have a snake case name

fn main() {}
