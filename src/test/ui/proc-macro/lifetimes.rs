// aux-build:lifetimes.rs

#![feature(proc_macro_hygiene)]

extern crate lifetimes;

use lifetimes::*;

type A = single_quote_alone!(); //~ ERROR expected type, found `'`

fn main() {}
