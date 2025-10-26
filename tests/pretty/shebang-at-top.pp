#!/usr/bin/env rust
#![feature(prelude_import)]
#![no_std]
#[macro_use]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-mode:expanded
//@ pp-exact:shebang-at-top.pp
//@ pretty-compare-only

fn main() {}
