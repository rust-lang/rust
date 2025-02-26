// Compiler:
//
// Run-time:
//   status: 1

#![feature(auto_traits, lang_items, no_core)]
#![allow(internal_features)]

#![no_std]
#![no_core]
#![no_main]

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
    1
}
