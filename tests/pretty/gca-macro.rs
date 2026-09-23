//@ pretty-mode:expanded
//@ pp-exact:gca-macro.pp
#![feature(min_generic_const_args)]

use std::gca;

fn f<const N: usize>() {}

fn main() {
    f::<gca!(2)>();
    f::<{ gca!(2) }>();
}
