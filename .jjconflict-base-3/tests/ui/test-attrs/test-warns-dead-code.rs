//@ compile-flags: --test

#![deny(dead_code)]

fn dead() {} //~ error: function `dead` is never used

fn main() {}
