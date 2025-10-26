#![feature(prelude_import)]
#![no_std]
#[macro_use]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:issue-12590-c.pp

// The next line should be expanded

#[path = "issue-12590-b.rs"]
mod issue_12590_b {

    fn b() {}
    fn main() {}
}
fn main() {}
