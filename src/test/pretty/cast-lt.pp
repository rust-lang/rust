#![feature(prelude_import)]
#![no_std]
#[prelude_import]
use ::std::prelude::v1::*;
#[macro_use]
extern crate std;
// pretty-compare-only
// pretty-mode:expanded
// pp-exact:cast-lt.pp

macro_rules! negative(( $ e : expr ) => { $ e < 0 });

fn main() { (1 as i32) < 0; }
