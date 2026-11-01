//! Test that Miri is able to run no_core programs.
//! This ensures that we don't depend on any paths from core when no_core is set.

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

#[no_mangle]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    0
}
