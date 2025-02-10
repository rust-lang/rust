// Compiler:
//
// Run-time:
//   status: 2

#![feature(auto_traits, lang_items, no_core, intrinsics)]
#![allow(internal_features)]

#![no_std]
#![no_core]
#![no_main]

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

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    unsafe {
        libc::exit(2);
    }
    0
}
