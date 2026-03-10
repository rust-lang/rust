#![deny = "foo"] //~ ERROR malformed `deny` attribute input
#![allow(bar = "baz")]
//~^ ERROR malformed `allow` attribute input [E0539]
fn main() { }
