// aux-build:test-macros.rs

extern crate test_macros;

use test_macros::empty_derive;
//~^ ERROR: unresolved import `test_macros::empty_derive`

fn main() {}
