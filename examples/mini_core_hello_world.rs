// Adapted from https://github.com/sunfishcode/mir2cranelift/blob/master/rust-examples/nocore-hello-world.rs

#![feature(no_core, unboxed_closures, start, lang_items)]
#![no_core]
#![allow(dead_code)]

extern crate mini_core;

use mini_core::*;

#[link(name = "c")]
extern "C" {}

extern "C" {
    fn puts(s: *const u8);
}

#[lang = "termination"]
trait Termination {
    fn report(self) -> i32;
}

impl Termination for () {
    fn report(self) -> i32 {
        unsafe {
            NUM = 6 * 7 + 1 + (1u8 == 1u8) as u8; // 44
            *NUM_REF as i32
        }
    }
}

#[lang = "start"]
fn start<T: Termination + 'static>(
    main: fn() -> T,
    _argc: isize,
    _argv: *const *const u8,
) -> isize {
    main().report() as isize
}

static mut NUM: u8 = 6 * 7;
static NUM_REF: &'static u8 = unsafe { &NUM };

fn main() {
    unsafe {
        let (ptr, _): (*const u8, usize) = intrinsics::transmute("Hello!\0");
        puts(ptr);
    }

    //panic(&("panic msg", "abc.rs", 0, 43));
}
