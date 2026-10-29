//@ aux-build: non_const_impl.rs
#![crate_type = "lib"]

extern crate non_const_impl;

use non_const_impl::X;

const _: () = {
    let x = X;
    x == x;
    //~^ ERROR: cannot call non-const operator in constants
    //~| NOTE: impl defined here, but it is not `const`
    //~| NOTE: limited to constant functions
};
