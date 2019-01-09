// compile-flags: --test

#![deny(dead_code)]

fn dead() {} //~ error: function is never used: `dead`

fn main() {}
