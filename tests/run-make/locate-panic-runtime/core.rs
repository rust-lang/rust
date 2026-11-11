// We are core.
#![feature(lang_items, no_core)]
#![allow(internal_features)]
#![no_std]
#![no_core]
#![crate_type = "rlib"]

#[lang = "panic_info"]
pub struct PanicInfo {}

#[lang = "copy"]
pub trait Copy: Sized {}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}
