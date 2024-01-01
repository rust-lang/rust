// compile-flags: --edition=2018
// build-pass

#![warn(unused)]

macro_rules! regex {
    //~^ WARN unused macro definition
    () => {};
}

#[allow(dead_code)]
use regex;
//~^ WARN unused import

fn main() {}
