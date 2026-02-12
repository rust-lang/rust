// Regression test for issues #152004 and #151124.

#![deny(unused)]

mod m {
    pub struct S {}
}

use m::*; //~ ERROR unused import: `m::*`
pub use m::*;

fn main() {}
