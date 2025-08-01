#![feature(prelude_import)]
#![no_std]
#[macro_use]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:if-else.pp

fn f(x: u32, y: u32) {
    let mut a = 0;
    if x > y { a = 1; } else { a = 2; }

    if x < 1 {
        a = 1;
    } else if x < 2 {
        a = 2;
    } else if x < 3 { a = 3; } else if x < 4 { a = 4; } else { a = 5; }

    if x < y {
        a += 1;
        a += 1;
        a += 1;
    } else {
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
        a += 1;
    }

    if x < 1 {
        if x < 2 {
            if x < 3 {
                a += 1;
            } else if x < 4 { a += 1; if x < 5 { a += 1; } }
        } else if x < 6 { a += 1; }
    }
}

fn main() { f(3, 4); }
