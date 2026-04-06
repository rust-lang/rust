//@ aux-build: non_const_impl.rs
#![crate_type = "lib"]

extern crate non_const_impl;

use non_const_impl::X;

const _: () = {
    let x = X;
    x == x;
    //~^ ERROR: their message
    //~| NOTE: their label
    //~| NOTE: trait `PartialEq` is implemented but not `const`
    //~| NOTE: their note
    //~| NOTE: their other note
};
