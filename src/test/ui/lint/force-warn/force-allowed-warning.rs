// compile-flags: --force-warns dead_code
// check-pass

#![allow(dead_code)]

fn dead_function() {}
//~^ WARN function is never used

fn main() {}
