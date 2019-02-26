#![deny = "foo"] //~ ERROR attribute must be of the form
#![allow(bar = "baz")] //~ ERROR malformed lint attribute

fn main() { }
