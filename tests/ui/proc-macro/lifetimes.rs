//@ proc-macro: lifetimes.rs

extern crate lifetimes;

use lifetimes::*;

type A = single_quote_alone!(); //~ ERROR expected type, found `'`

fn main() {}
