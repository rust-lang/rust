//@ compile-flags: --test

// A `fn main` demoted by an explicit `#[rustc_main]` on another function is dead code and
// must be flagged under `--test` just like in a normal build.

#![allow(unused_variables)]
#![deny(dead_code)]
#![feature(rustc_attrs)]

fn dead_fn() {} //~ ERROR: function `dead_fn` is never used

fn used_fn() {}

#[rustc_main]
fn actual_main() {
    used_fn();
}

// this is not main
fn main() { //~ ERROR: function `main` is never used
    dead_fn();
}
