// Test that we DO warn when lifetime name is not used at all.

#![deny(unused_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

fn d<'a>() { } //~ ERROR `'a` never used

fn main() { }
