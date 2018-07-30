// Adapted from https://github.com/sunfishcode/mir2cranelift/blob/master/rust-examples/nocore-hello-world.rs

#![feature(no_core, unboxed_closures, start)]
#![no_core]
#![allow(dead_code)]

extern crate mini_core;

use mini_core::*;

#[link(name = "c")]
extern "C" {}

extern "C" {
    fn puts(s: *const u8);
}

#[start]
fn main(i: isize, _: *const *const u8) -> isize {
    unsafe {
        let (ptr, _): (*const u8, usize) = intrinsics::transmute("Hello!\0");
        puts(ptr);
    }
    0
}
