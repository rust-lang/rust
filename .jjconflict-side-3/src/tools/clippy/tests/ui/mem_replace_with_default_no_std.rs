//@aux-build:proc_macros.rs
#![warn(clippy::mem_replace_with_default)]
#![no_std]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

use core::mem;

fn it_works() {
    let mut refstr = "hello";
    let _ = mem::replace(&mut refstr, "");
    //~^ mem_replace_with_default

    let mut slice: &[i32] = &[1, 2, 3];
    let _ = mem::replace(&mut slice, &[]);
    //~^ mem_replace_with_default
}

#[inline_macros]
fn macros(mut refstr: &str) {
    let _ = inline!(mem::replace(&mut $refstr, ""));
    //~^ mem_replace_with_default
    let _ = external!(mem::replace(&mut $refstr, ""));
}
