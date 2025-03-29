#!/usr/bin/env rust
#![feature(prelude_import)]
#![no_std]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
//@ pretty-mode:expanded
//@ pp-exact:shebang-at-top.pp
//@ pretty-compare-only

fn main() {}
