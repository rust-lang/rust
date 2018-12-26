// aux-build:negative-token.rs

#![feature(proc_macro_hygiene)]

extern crate negative_token;

use negative_token::*;

fn main() {
    assert_eq!(-1, neg_one!());
    assert_eq!(-1.0, neg_one_float!());
}
