// Adapted from https://github.com/sunfishcode/mir2cranelift/blob/master/rust-examples/nocore-hello-world.rs

#![feature(no_core, unboxed_closures, start, lang_items)]
#![no_core]
#![no_main]
#![allow(dead_code)]

extern crate mini_core;

use mini_core::*;

#[link(name = "c")]
extern "C" {}

extern "C" {
    fn puts(s: *const u8);
}

static NUM: u8 = 6 * 7;

#[lang = "start"]
fn start(_main: *const u8, i: isize, _: *const *const u8) -> isize {
    unsafe {
        let (ptr, _): (*const u8, usize) = intrinsics::transmute("Hello!\0");
        puts(ptr);
    }

    NUM as isize
}
