// Test that when we call a function with #[rustc_panic_entrypoint], we encounter an abort and do
// not actually execute the function. Immediate-aborting panics are implemented by a rewrite of such
// functions, so this tests that the rewrite works with Miri.

//@compile-flags: -Cpanic=immediate-abort -Zunstable-options --target=x86_64-unknown-none

#![no_std]
#![no_core]
#![no_main]
#![feature(rustc_attrs, no_core, lang_items, intrinsics)]
#![allow(internal_features)]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[rustc_nounwind]
#[rustc_intrinsic]
#[lang = "abort_intrinsic"]
fn abort() -> !;

#[no_mangle]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    panic_entrypoint(); //~ ERROR: abnormal termination: the program aborted execution
    0
}

#[rustc_panic_entrypoint]
fn panic_entrypoint() {}
