#![feature(prelude_import)]
#![no_std]
//@ pretty-mode:expanded
//@ pp-exact:gca-macro.pp
#![feature(gca_min_const_items)]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

use std::gca;

fn f<const N : usize>() {}

fn main() { f::<core::gca! (2)>(); f::<{ core::gca! (2) }>(); }
