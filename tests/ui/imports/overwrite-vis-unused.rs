// Regression test for issues #152004 and #151124.
//@ check-pass
#![deny(unused)]

mod m {
    pub struct S {}
}

use m::*;
pub use m::*;

fn main() {}
