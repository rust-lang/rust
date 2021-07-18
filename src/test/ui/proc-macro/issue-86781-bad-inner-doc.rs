// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

//! Inner doc comment
//~^ ERROR expected outer doc comment
#[derive(Empty)]
pub struct Foo;

fn main() {}
