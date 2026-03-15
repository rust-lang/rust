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
#[diagnostic::on_unimplemented(
    message = "values of type `{Self}` may or may not have a size",
    label = "may or may not have a known size"
)]
pub trait PointeeSized {}

#[lang = "meta_sized"]
#[diagnostic::on_unimplemented(
    message = "the size for values of type `{Self}` cannot be known",
    label = "doesn't have a known size"
)]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
#[diagnostic::on_unimplemented(
    message = "the size for values of type `{Self}` cannot be known at compilation time",
    label = "doesn't have a size known at compile-time"
)]
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
