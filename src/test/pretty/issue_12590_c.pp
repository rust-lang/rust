#![feature(prelude_import)]
#![no_std]
#[prelude_import]
use ::std::prelude::v1::*;
#[macro_use]
extern crate std;
// pretty-compare-only
// pretty-mode:expanded
// pp-exact:issue_12590_c.pp

// The next line should be expanded

mod issue_12590_b {

    fn b() { }
    fn main() { }
}
fn main() { }
