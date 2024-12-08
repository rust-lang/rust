// Compiler:
//
// Run-time:
//   status: 0

#![feature(auto_traits, lang_items, no_core, start)]
#![allow(internal_features)]

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

/*
 * Code
 */

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    0
}
