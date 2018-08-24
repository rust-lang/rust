#![deny = "foo"] //~ ERROR malformed lint attribute
#![allow(bar = "baz")] //~ ERROR malformed lint attribute

fn main() { }
