#![deny = "foo"] //~ ERROR malformed `deny` attribute input
#![allow(bar = "baz")]
//~^ ERROR malformed
//~| ERROR malformed
//~| ERROR malformed
//~| WARN  unused attribute

fn main() {}
