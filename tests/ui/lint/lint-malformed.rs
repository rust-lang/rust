#![deny = "foo"] //~ ERROR malformed `deny` attribute input
#![allow(bar = "baz")] //~ ERROR malformed
                       //~| ERROR malformed
                       //~| ERROR malformed
                       //~| ERROR malformed
                       //~| ERROR malformed
                       //~| ERROR malformed
fn main() { }
