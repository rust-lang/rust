// compile-flags: --force-warns dead_code -Zunstable-options
// check-pass

#![allow(dead_code)]

fn dead_function() {}
//~^ WARN function is never used

fn main() {}
