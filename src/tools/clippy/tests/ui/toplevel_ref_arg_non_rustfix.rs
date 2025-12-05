//@aux-build:proc_macros.rs

#![warn(clippy::toplevel_ref_arg)]
#![allow(unused)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

fn the_answer(ref mut x: u8) {
    //~^ toplevel_ref_arg
    *x = 42;
}

#[inline_macros]
fn main() {
    let mut x = 0;
    the_answer(x);

    // lint in macro
    inline! {
        fn fun_example(ref _x: usize) {}
        //~^ toplevel_ref_arg
    }

    // do not lint in external macro
    external! {
        fn fun_example2(ref _x: usize) {}
    }
}
