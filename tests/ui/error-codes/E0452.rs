#![allow(foo = "")]
//~^ ERROR malformed
//~| ERROR malformed
//~| ERROR malformed
//~| WARN unused attribute

fn main() {}
