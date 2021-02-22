// aux-build:test-macros.rs

#![dummy] //~ ERROR cannot find attribute `dummy` in this scope

#[macro_use]
extern crate test_macros;

#[derive(Empty)] //~ ERROR cannot determine resolution for the attribute macro `derive`
#[empty_helper] //~ WARN derive helper attribute is used before it is introduced
                //~| WARN this was previously accepted
struct Foo {}

fn main() {}
