//@ run-pass
#![allow(non_snake_case)]

//@ aux-build:issue-14421.rs


extern crate issue_14421 as bug_lib;

use bug_lib::B;
use bug_lib::make;

pub fn main() {
    let mut an_A: B = make();
    an_A.foo();
}
