// aux-build:lifetimes.rs

#![feature(proc_macro_non_items)]

extern crate lifetimes;

use lifetimes::*;

type A = single_quote_alone!(); //~ ERROR expected type, found `'`
