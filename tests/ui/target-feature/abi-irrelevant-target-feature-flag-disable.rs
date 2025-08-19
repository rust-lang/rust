//! `x87` is a required target feature on some x86 targets, but not on this one as this one
//! uses soft-floats. So ensure disabling the target feature here (which is a NOP) does
//! not trigger a warning.
//@ compile-flags: --target=x86_64-unknown-none --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-x87
//@ build-pass
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

//~? WARN unstable feature specified for `-Ctarget-feature`: `x87`
