// Test that we DO warn when lifetime name is not used at all.

#![deny(unused_lifetimes)]
#![allow(dead_code, unused_variables)]

struct Foo {}

impl<'a> Foo {} //~ ERROR `'a` never used

fn main() {}
