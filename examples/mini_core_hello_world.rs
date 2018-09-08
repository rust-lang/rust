// Adapted from https://github.com/sunfishcode/mir2cranelift/blob/master/rust-examples/nocore-hello-world.rs

#![feature(no_core, unboxed_closures, start, lang_items, box_syntax)]
#![no_core]
#![allow(dead_code)]

extern crate mini_core;

use mini_core::*;

#[link(name = "c")]
extern "C" {
    fn puts(s: *const u8);
}

unsafe extern "C" fn my_puts(s: *const u8) {
    puts(s);
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
        let slice: &[u8] = b"Hello\0" as &[u8; 6];
        if intrinsics::size_of_val(slice) as u8 != 6 {
            panic(&("eji", "frjio", 0, 0));
        };
        let ptr: *const u8 = slice as *const [u8] as *const u8;
        let world = box "World!\0";
        puts(ptr);
        puts(*world as *const str as *const u8);
    }

    //panic(&("panic msg", "abc.rs", 0, 43));
}
