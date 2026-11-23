#![deny = "foo"] //~ ERROR malformed `deny` attribute input
#![allow(bar = "baz")] //~ ERROR malformed

fn main() {}
