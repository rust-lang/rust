// Compiler:
//
// Run-time:
//   status: 2

#![feature(auto_traits, lang_items, no_core, start, intrinsics)]

#![no_std]
#![no_core]

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn exit(status: i32);
    }
}

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

/*
 * Code
 */

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        libc::exit(2);
    }
    0
}
