#![allow(unused_macros)]

#[rustc_allow_const_fn_unstable()] //~ ERROR rustc_allow_const_fn_unstable side-steps
const fn foo() { }

fn main() {}
