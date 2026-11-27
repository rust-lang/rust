//@ pretty-mode:expanded
//@ pp-exact:gca-macro.pp
#![feature(gca_min_const_items)]

use std::gca;

fn f<const N: usize>() {}

fn main() {
    f::<gca!(2)>();
    f::<{ gca!(2) }>();
}
