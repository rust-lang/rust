#![feature(prelude_import)]
#![no_std]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:dollar-crate.pp

fn main() { { ::std::io::_print(format_args!("rust\n")); }; }
