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

static mut NUM: u8 = 6 * 7;
static NUM_REF: &'static u8 = unsafe { &NUM };

#[lang = "start"]
fn start(_main: *const u8, i: isize, _: *const *const u8) -> isize {
    unsafe {
        let (ptr, _): (*const u8, usize) = intrinsics::transmute("Hello!\0");
        puts(ptr);
    }

    unsafe {
        NUM = 43;
        *NUM_REF as isize
    }
}
