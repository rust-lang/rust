//@ known-bug: #116979
//@ compile-flags: -Csymbol-mangling-version=v0
//@ needs-rustc-debug-assertions

#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Display;

pub fn require_dyn_star_display(_: dyn* Display) {}

fn main() {
    require_dyn_star_display(1usize);
}
