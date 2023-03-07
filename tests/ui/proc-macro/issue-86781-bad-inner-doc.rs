// aux-build:test-macros.rs
// run-rustfix

#[macro_use]
extern crate test_macros;

//! Inner doc comment
//~^ ERROR expected outer doc comment
#[derive(Empty)]
pub struct Foo; //~ NOTE the inner doc comment doesn't annotate this struct

fn main() {}
