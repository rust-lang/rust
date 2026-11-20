#![allow(unused_macros)]

#[rustc_allow_const_fn_unstable()] //~ ERROR use of an internal attribute
const fn foo() { }

fn main() {}
