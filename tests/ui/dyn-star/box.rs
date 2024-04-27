//@ run-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] compile-flags: -C opt-level=0
//@[next] compile-flags: -Znext-solver -C opt-level=0

#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Display;

fn make_dyn_star() -> dyn* Display {
    Box::new(42) as dyn* Display
}

fn main() {
    let x = make_dyn_star();

    println!("{x}");
}
