// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 3

#![feature(auto_traits, lang_items, no_core, start, intrinsics)]

#![no_std]
#![no_core]

/*
 * Core
 */

// Because we don't have core yet.
#[lang = "sized"]
pub trait Sized {}

#[lang = "copy"]
trait Copy {
}

impl Copy for isize {}

#[lang = "receiver"]
trait Receiver {
}

#[lang = "freeze"]
pub(crate) unsafe auto trait Freeze {}

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn printf(format: *const i8, ...) -> i32;
    }
}

/*
 * Code
 */

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    let test: (isize, isize, isize) = (3, 1, 4);
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, test.0);
    }
    0
}
