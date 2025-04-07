// Compiler:
//
// Run-time:
//   status: signal

#![feature(auto_traits, lang_items, no_core, intrinsics, rustc_attrs)]
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

mod intrinsics {
    use super::Sized;

    #[rustc_nounwind]
    #[rustc_intrinsic]
    pub fn abort() -> !;
}

/*
 * Code
 */

fn test_fail() -> ! {
    unsafe { intrinsics::abort() };
}

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    test_fail();
}
