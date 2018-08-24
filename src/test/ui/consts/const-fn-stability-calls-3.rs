// Test use of const fn from another crate without a feature gate.

#![feature(rustc_attrs)]
#![allow(unused_variables)]

// aux-build:const_fn_lib.rs

extern crate const_fn_lib;

use const_fn_lib::foo;

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let x = foo(); // use outside a constant is ok
}
