#![deny = "foo"] //~ ERROR malformed `deny` attribute input
#![allow(bar = "baz")] //~ ERROR expected a lint
                       //~| ERROR expected a lint
                       //~| ERROR expected a lint
                       //~| ERROR expected a lint
                       //~| ERROR expected a lint
                       //~| ERROR expected a lint
fn main() { }
