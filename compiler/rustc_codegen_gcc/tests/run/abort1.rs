// Compiler:
//
// Run-time:
//   status: signal

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

mod intrinsics {
    use super::Sized;

    extern "rust-intrinsic" {
        pub fn abort() -> !;
    }
}

/*
 * Code
 */

fn test_fail() -> ! {
    unsafe { intrinsics::abort() };
}

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    test_fail();
}
