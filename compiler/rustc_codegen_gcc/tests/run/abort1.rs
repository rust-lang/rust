// Compiler:
//
// Run-time:
//   status: signal

#![feature(auto_traits, lang_items, no_core, start, intrinsics, rustc_attrs)]
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

mod intrinsics {
    use super::Sized;

    #[rustc_nounwind]
    #[rustc_intrinsic]
    #[rustc_intrinsic_must_be_overridden]
    pub fn abort() -> ! {
        loop {}
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
