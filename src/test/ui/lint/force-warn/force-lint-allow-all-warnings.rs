// compile-flags: --force-warns dead_code
// check-pass

#![allow(warnings)]

fn dead_function() {}
//~^ WARN function is never used

fn main() {}
