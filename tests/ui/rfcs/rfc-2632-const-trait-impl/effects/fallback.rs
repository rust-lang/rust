//@ check-pass
//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(effects)]

pub const fn owo() {}

fn main() {
    // make sure falling back ty/int vars doesn't cause const fallback to be skipped...
    // See issue: 115791.
    let _ = 1;
    if false {
        let x = panic!();
    }

    let _ = owo;
}
